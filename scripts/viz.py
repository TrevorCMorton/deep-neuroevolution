import click


@click.command()
@click.argument('env_ids')
@click.argument('policy_directory')
@click.option('--record', is_flag=True)
@click.option('--stochastic', is_flag=True)
@click.option('--extra_kwargs')
def main(env_ids, policy_directory, record, stochastic, extra_kwargs):
    import gym
    from gym import wrappers
    import tensorflow as tf
    from es_distributed.policies import MujocoPolicy, ESAtariPolicy, GAAtariPolicy
    from es_distributed.atari_wrappers import ScaledFloatFrame, wrap_deepmind
    from es_distributed.es import get_ref_batch
    import es_distributed.ns as ns
    import numpy as np
    import os

    env_ids = env_ids.split(' ')

    is_atari_policy = "NoFrameskip" in env_ids[0]

    files = 0

    for policy_name in os.listdir(policy_directory):
        files += 1
        policy_file = "%s/%s" % (policy_directory, policy_name)
        pid = os.fork()
        if (pid == 0):
            env = []
            for i in range(0, len(env_ids)):
                env.append(gym.make(env_ids[i]))
                if env_ids[i].endswith('NoFrameskip-v4'):
                    env[i] = wrap_deepmind(env[i])

            if extra_kwargs:
                import json
                extra_kwargs = json.loads(extra_kwargs)

            with tf.Session():
                if is_atari_policy:
                    pi = GAAtariPolicy.Load(policy_file, extra_kwargs=extra_kwargs)
                    if pi.needs_ref_batch:
                        pi.set_ref_batch(get_ref_batch(env[0], batch_size=128))
                else:
                    pi = MujocoPolicy.Load(policy_file, extra_kwargs=extra_kwargs)

                while True:
                    if is_atari_policy:
                        rews, t, novelty_vector = pi.rollout(env, render=True, random_stream=np.random if stochastic else None)

    for i in range(0, files):
        os.wait()


if __name__ == '__main__':
    main()
