from run import *
from agent import AgentDQN, AgentUADQN
from StockTrading import StockTradingEnv, check_stock_trading_env
import yfinance as yf
from stockstats import StockDataFrame as Sdf

def setup_args(agent, agent_inputs, env, env_eval):
    # Agent
    args = Arguments(if_on_policy=False)
    args.agent_specific_inputs = agent_inputs
    args.agent = agent


    args.env = env
    args.env_eval = env_eval

    args.env.target_reward = 3
    args.env_eval.target_reward = 3

    # Hyperparameters
    args.gamma = args.env.gamma
    args.break_step = 200
    args.net_dim = 2 ** 9
    args.max_step = args.env.max_step
    args.max_memo = args.max_step * 4
    args.batch_size = 32
    args.repeat_times = 2 # repeat_times * target_step == number of times we update before training
    args.eval_gap = 2 ** 4
    args.eval_times1 = 2 ** 3
    args.eval_times2 = 2 ** 5
    args.if_allow_break = True
    args.rollout_num = 2 # the number of rollout workers (larger is not always faster)
    args.target_step = 50 # number of exploration steps before training

    return args

def env_setup(ticker='TARA'):
    # Environment
    tickers = ticker  # finrl.config.NAS_74_TICKER

    tech_indicator_list = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
    'close_30_sma', 'close_60_sma']  # finrl.config.TECHNICAL_INDICATORS_LIST
    
    gamma = 0.99
    max_stock = 1
    initial_capital = 1e6
    initial_stocks = 100
    buy_cost_pct = 1e-3
    sell_cost_pct = 1e-3
    start_date = '2008-01-01'
    start_eval_date = '2016-01-01'
    end_eval_date = '2021-01-01'

    env = StockTradingEnv('./'+ticker+'/', gamma, max_stock, initial_capital, buy_cost_pct, 
                          sell_cost_pct, start_date, start_eval_date, 
                          end_eval_date, tickers, tech_indicator_list, 
                          initial_stocks, if_eval=False, if_save=False, if_save=True)
    env_eval = StockTradingEnv('./'+ticker+'/', gamma, max_stock, initial_capital, buy_cost_pct, 
                              sell_cost_pct, start_date, start_eval_date, 
                              end_eval_date, tickers, tech_indicator_list, 
                              initial_stocks, if_eval=True, if_save=False, if_save=True)
    return env, env_eval

def UADQN_setup(env, env_eval):
    agent = AgentUADQN()
    agent_inputs = {"kappa": 1, "prior":0.01, "aleatoric_penalty":0.5, "n_quantiles":200}
    return setup_args(agent, agent_inputs, env, env_eval)

def DQN_setup(env, env_eval):
    agent = AgentDQN()
    agent_inputs = {"explore_rate": 0.1, "turbulence_threshold":300}
    return setup_args(agent, agent_inputs, env, env_eval)

def baseline_model(env, _torch):
    state, fti = env.reset()
    episode_returns = list()
    print('The initial captial is {}'.format(env.initial_capital))
    print('The initial number of stocks is {}'.format(env.initial_stocks))
    for i in range(env.max_step):
        state, reward, done, fti, _ = env.step(2)

        total_asset = env.amount + env.price_ary[env.day] * env.stocks
        episode_return = total_asset / env.initial_total_asset
        episode_returns.append(episode_return)
        if done:
            break
    
    cwd = f'Baseline/{env.env_name}'
    os.makedirs(cwd, exist_ok=True)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(episode_returns)
    plt.grid()
    plt.title('cumulative return over time')
    plt.xlabel('day')
    plt.ylabel('cumulative return as fraction of initial asset')
    plt.savefig(f'{cwd}/cumulative_return.jpg')
    np.save('%s/cumulative_return.npy' % cwd, episode_returns)
    plt.close('all')

    print('BASELINE MODEL RETURN: {}\n'.format(episode_returns[-1]))


if __name__ == '__main__':
    tickers = ['TARA', 'AAPL']
    for stock_ticker in tickers:
        print('------- Begin Experiment for {} --------\n'.format(stock_ticker))
        env, env_eval = env_setup(stock_ticker)
        print('-'*50)
        print('RUNNING BASELINE MODEL')
        baseline_model(env_eval, torch)
        print('-'*50)
        print('RUNNING DQN MODEL')
        args = DQN_setup(env, env_eval)
        train_and_evaluate(args)
        returns = args.env_eval.draw_cumulative_return(args, torch)
        print('DQN MODEL RETURN: {}\n'.format(returns[-1]))
        returns = args.env_eval.draw_cumulative_return_while_learning(args, torch)
        print('DQN MODEL RETURN WHILE LEARNING: {}\n'.format(returns[-1]))
        print('-'*50)
        print('RUNNING UADQN MODEL')
        args = UADQN_setup(env, env_eval)
        train_and_evaluate(args)
        returns = args.env_eval.draw_cumulative_return(args, torch)
        print('UADQN MODEL RETURN: {}\n'.format(returns[-1]))
        returns = args.env_eval.draw_cumulative_return_while_learning(args, torch)
        print('UADQN MODEL RETURN WHILE LEARNING: {}\n'.format(returns[-1]))
        print('-'*50)
    
