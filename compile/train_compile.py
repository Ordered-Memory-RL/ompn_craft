import torch
import gym
import gym_psketch
from gym_psketch import DictList, ID2SKETCHS
import compile
from gym_psketch.evaluate import f1, get_subtask_seq, get_ta_lines
from absl import flags, logging
import numpy as np
import os
from tensorboardX import SummaryWriter

FLAGS = flags.FLAGS
flags.DEFINE_integer('compile_train_steps', default=4000, help='train steps')
flags.DEFINE_integer('compile_eval_freq', default=20, help='evaluation frequency')
flags.DEFINE_float('compile_lr', default=0.001, help='learning rate')
flags.DEFINE_integer('compile_batch_size', default=256, help='learning rate')
flags.DEFINE_integer('compile_max_segs', default=4, help='num of segment')
flags.DEFINE_float('compile_beta_z', default=0.1, help='weight Z kl loss')
flags.DEFINE_float('compile_beta_b', default=0.1, help='weight b kl loss')
flags.DEFINE_float('compile_prior_rate', default=3, help='possion distribution. avg length of each seg')
flags.DEFINE_enum('compile_latent', enum_values=['gaussian', 'concrete'], default='gaussian',
                  help='Latent type')


def main(training_folder):
    logging.info('Start compile...')
    first_env = gym.make(FLAGS.envs[0])
    n_feature, action_size = first_env.n_features, first_env.n_actions
    model = compile.CompILE(vec_size=n_feature,
                            hidden_size=FLAGS.hidden_size,
                            action_size=action_size,
                            env_arch=FLAGS.env_arch,
                            max_num_segments=FLAGS.compile_max_segs,
                            latent_dist=FLAGS.compile_latent,
                            beta_b=FLAGS.compile_beta_b,
                            beta_z=FLAGS.compile_beta_z,
                            prior_rate=FLAGS.compile_prior_rate)
    if FLAGS.cuda:
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.compile_lr)

    # Dataset
    dataloader = compile.Dataloader(env_names=FLAGS.envs,
                                    val_ratio=0.2,
                                    use_bot=FLAGS.il_demo_from_model)
    train_steps = 0
    writer = SummaryWriter(training_folder)
    train_iter = dataloader.train_iter(batch_size=FLAGS.compile_batch_size)
    nb_frames = 0
    curr_best = np.inf
    train_stats = DictList()
    while True:
        if train_steps > FLAGS.compile_train_steps:
            logging.info('Reaching maximum steps')
            break

        if train_steps % FLAGS.compile_eval_freq == 0:
            # Testing
            val_metrics = {}
            model.eval()
            for env_name in FLAGS.envs:
                val_metrics[env_name] = DictList()
                val_iter = dataloader.val_iter(batch_size=FLAGS.compile_batch_size, env_names=[env_name])
                for val_batch, val_lengths in val_iter:
                    if FLAGS.cuda:
                        val_batch.apply(lambda _t: _t.cuda())
                        val_lengths = val_lengths.cuda()
                    with torch.no_grad():
                        val_outputs, extra_info = model.forward(val_batch, val_lengths)
                    val_metrics[env_name].append(val_outputs)

                # Parsing
                total_lengths = 0
                total_task_corrects = 0
                val_iter = dataloader.val_iter(batch_size=FLAGS.eval_episodes, env_names=[env_name], shuffle=True)
                val_batch, val_lengths = val_iter.__next__()
                if FLAGS.cuda:
                    val_batch.apply(lambda _t: _t.cuda())
                    val_lengths = val_lengths.cuda()
                with torch.no_grad():
                    val_outputs, extra_info = model.forward(val_batch, val_lengths)
                seg = torch.stack(extra_info['segment'], dim=1).argmax(-1)
                for _action, _seg, _length, _env_id in zip(val_batch.action, seg, val_lengths, val_batch.env_id):
                    __action = _action[:_length]
                    use_id = ((__action == gym_psketch.Actions.USE.value).nonzero()).view(-1).cpu()
                    _seg = _seg[_seg.sort()[1]].cpu()

                    # Remove the last one because too trivial
                    val_metrics[env_name].append({'f1_tol0': f1(use_id, _seg, 0),
                                                  'f1_tol1': f1(use_id, _seg, 1),
                                                  'f1_tol2': f1(use_id, _seg, 2)})

                    # subtask
                    total_lengths += _length.item()
                    gt_subtask = get_subtask_seq(__action, subtask=ID2SKETCHS[_env_id[0].item()])
                    decoded_subtask = get_subtask_seq(__action, subtask=ID2SKETCHS[_env_id[0].item()],
                                                      use_ids=_seg)
                    total_task_corrects += (gt_subtask == decoded_subtask).float().sum()

                # record task acc
                val_metrics[env_name].task_acc = total_task_corrects / total_lengths

                # Print parsing result
                pred_bound = torch.zeros_like(__action)
                pred_bound[_seg.clamp(0, _length - 1)] = 1
                lines = get_ta_lines(action=__action, decoded_subtask=decoded_subtask,
                                     gt_subtask=gt_subtask)
                logging.info('\n'.join(lines))
                val_metrics[env_name].apply(lambda _t: torch.tensor(_t).float().mean().item())

            # Logger
            for env_name, metric in val_metrics.items():
                line = ['[VALID][{}] steps={}'.format(env_name, train_steps)]
                for k, v in metric.items():
                    line.append('{}: {:.4f}'.format(k, v))
                logging.info('\t'.join(line))

            mean_val_metric = DictList()
            for metric in val_metrics.values():
                mean_val_metric.append(metric)
            mean_val_metric.apply(lambda t: torch.mean(torch.tensor(t)))
            for k, v in mean_val_metric.items():
                writer.add_scalar('val/' + k, v.item(),nb_frames)
            writer.flush()

            avg_loss = [val_metrics[env_name].loss for env_name in val_metrics]
            avg_loss = np.mean(avg_loss)
            if avg_loss < curr_best:
                curr_best = avg_loss
                logging.info('Save Best with loss: {}'.format(avg_loss))
                # Save the checkpoint
                with open(os.path.join(training_folder, 'bot_best.pkl'), 'wb') as f:
                    torch.save(model, f)

        model.train()
        train_batch, train_lengths = train_iter.__next__()
        if FLAGS.cuda:
            train_batch.apply(lambda _t: _t.cuda())
            train_lengths = train_lengths.cuda()
        train_outputs, _ = model.forward(train_batch, train_lengths)

        optimizer.zero_grad()
        train_outputs['loss'].backward()
        optimizer.step()
        train_steps += 1
        nb_frames += train_lengths.sum().item()

        train_outputs = DictList(train_outputs)
        train_outputs.apply(lambda _t: _t.item())
        train_stats.append(train_outputs)

        if train_steps % FLAGS.compile_eval_freq == 0:
            train_stats.apply(lambda _tensors: np.mean(_tensors))
            logger_str = ['[TRAIN] steps={}'.format(train_steps)]
            for k, v in train_stats.items():
                logger_str.append("{}: {:.4f}".format(k, v))
                writer.add_scalar('train/' + k, v, global_step=nb_frames)
            logging.info('\t'.join(logger_str))
            train_stats = DictList()
            writer.flush()
