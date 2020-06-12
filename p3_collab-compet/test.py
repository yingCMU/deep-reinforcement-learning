import utilities
importlib.reload(utilities)

LAST_BEST_SCORE=float('-inf')

def ddpg_ma_version(agent, n_episodes, path_to_write_suffix,apply_noise=True, eval_mode=True, start_noise=10, noise_reduction=0.9999,min_noise=0.1, max_t=1000, last_best_score=LAST_BEST_SCORE, final_scores=[]):
    actors_path,critics_path = get_path(path_to_write_suffix)
    noise = start_noise if apply_noise else 0
    noise_reduction = noise_reduction
    print('--------start learning----------')
    num_episodes_over_criteria = 0
    last_100_scores_deque = deque(maxlen=100)
    for i_episode in range(1, n_episodes+1):
#         states = env.reset()[brain_name].vector_observations
        states = env.reset(train_mode=True)[brain_name].vector_observations
        agent.reset()
        ending_t = 0
        all_agent_scores = []
        reward_this_episode = np.zeros((1, 2))
        update_iteration = 0
        for t in range(max_t):
            # make add_noise=True when early in the training stage;
            # I changed it False at the end to verify the agent solves the environemnt
            actions_for_2 = agent.act(states, noise=noise)
            noise = max(noise*noise_reduction, min_noise)
            actions_array = torch.stack(actions_for_2).detach().cpu().numpy()
            actions_for_env = np.rollaxis(actions_array,1)
            env_info= env.step(actions_for_env)[brain_name]
            next_states_for_2 = env_info.vector_observations   # get the next state
            rewards_for_2 = env_info.rewards                   # get the reward
            done_for_2 = env_info.local_done
            # if model has not been improving, then update the networks 10 times after every 20 timesteps
            agent.add_to_memory(states, actions_array, rewards_for_2, next_states_for_2, done_for_2, i_episode)
            if not eval_mode:
                update_iteration = agent.learn_and_update(i_episode, t, start_update_episode=0, ts_per_update=1 , updates_per_ts=0.2, update_iteration=update_iteration)

            reward_this_episode += rewards_for_2
            states = next_states_for_2
            all_agent_scores.append(rewards_for_2)
            if np.any(done_for_2):
                ending_t=t
                break
        c = np.vstack(all_agent_scores)
        sum_all_agent_scores = c.sum(axis=0)
        episode_max_score =  np.max(sum_all_agent_scores)
        last_100_scores_deque.append(episode_max_score)
        last_100_avg_score =  np.mean(last_100_scores_deque)
        final_scores.append(episode_max_score)
        if last_best_score < episode_max_score:
            print('[...Saving model - episode {};T {} ; episode_max={:.2f}; last_100_avg_score={:.4f} path {}'.format(i_episode,ending_t, episode_max_score, last_100_avg_score,path_to_write_suffix))
            last_best_score = episode_max_score
            for agent_i in range(2):
                torch.save(agent.maddpg_agent[agent_i].actor_local.state_dict(), actors_path[agent_i])
                torch.save(agent.maddpg_agent[agent_i].critic_local.state_dict(), critics_path[agent_i])

        if last_100_avg_score >= 0.5:
            if num_episodes_over_criteria > 100:
                print('\rSolved environment after {} Episodes\t Last Max Agent Score: {:.2f}'.format(i_episode, episode_max_score))
                return final_scores


        if i_episode%100 == 0:
            print('\rEpisode {}; Ending_T {} ; episode_max: {:.12f} ; last_100_avg_score {:.4f}; Buffer {}.'.format(i_episode, ending_t, episode_max_score,last_100_avg_score, agent.memory.__len__()))
    print('----- Finish Training for {} Episodes, best score: {} | -------'.format(n_episodes, last_best_score))
    return final_scores
