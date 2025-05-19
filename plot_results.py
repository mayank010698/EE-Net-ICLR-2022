import numpy as np
import matplotlib.pyplot as plt


def get_mean_std(ress):
    return np.mean(ress, axis=0), np.std(ress, axis =0)
    

if __name__ == '__main__':    
    T = 10000
    x = range(T)
    plt.figure(figsize=(10, 6))

    ucb = np.load("results_new/LinUCB_regret_d_500_K_10_T_2000_cs_0.0_as_0.9_random_ball.npy")
    ucb = ucb[:,:2000]
    ucb_mean, ucb_std = get_mean_std(ucb)
    plt.plot(range(2000), ucb_mean, 'k-', color='blue',linewidth=2.0,linestyle=':', label = 'LinUCB')
    plt.fill_between(range(2000), ucb_mean-ucb_std, ucb_mean+ucb_std, facecolor='blue', alpha=0.2)

    ucb = np.load("/projects/illinois/eng/cs/arindamb/mayanks4/EE-Net-ICLR-2022/results_new/SketchLinUCB_regret_d_500_K_10_T_5000_cs_0.0_as_0.9_b_100_random_ball.npy")
    ucb = ucb[:,:5000]
    ucb_mean, ucb_std = get_mean_std(ucb)
    plt.plot(range(5000), ucb_mean, 'k-', color='red',linewidth=2.0,linestyle=':', label = 'SkLinUCB : b 100')
    plt.fill_between(range(5000), ucb_mean-ucb_std, ucb_mean+ucb_std, facecolor='red', alpha=0.2)

    ucb = np.load("results_new/SketchLinUCB_regret_d_500_K_10_T_5000_cs_0.9_as_0.0_b_500_random_ball.npy")
    ucb = ucb[:,:5000]
    ucb_mean, ucb_std = get_mean_std(ucb)
    plt.plot(range(5000), ucb_mean, 'k-', color='green',linewidth=2.0,linestyle=':', label = 'SkLinUCB: b 500')
    plt.fill_between(range(5000), ucb_mean-ucb_std, ucb_mean+ucb_std, facecolor='green', alpha=0.2)
    

    
    ucb = np.load("results_new/SketchLinUCB_regret_d_500_K_10_T_5000_cs_0.9_as_0.9_b_1000_random_ball.npy")
    ucb = ucb[:,:5000]
    print(ucb.shape)
    ucb_mean, ucb_std = get_mean_std(ucb)
    plt.plot(range(5000), ucb_mean, 'k-', color='orange',linewidth=2.0,linestyle=':', label = 'SkLinUCB: b 1000')
    plt.fill_between(range(5000), ucb_mean-ucb_std, ucb_mean+ucb_std, facecolor='orange', alpha=0.2)
    
    # ee = np.load('./results/eenet_results1.npy')
    # ee_mean, ee_std = get_mean_std(ee)
    # plt.plot(x, ee_mean, 'k-', color='red',linewidth=2.0,linestyle='-', label = 'EE-Net')
    # plt.fill_between(x, ee_mean-ee_std, ee_mean+ee_std, facecolor='red', alpha=0.2)

    
    # ucb = np.load("./results/NeuralUCB_regret.npy")
    # ucb_mean, ucb_std = get_mean_std(ucb)
    # plt.plot(x, ucb_mean, 'k-', color='blue',linewidth=2.0,linestyle=':', label = 'NeuralUCB')
    # plt.fill_between(x, ucb_mean-ucb_std, ucb_mean+ucb_std, facecolor='blue', alpha=0.2)

    # ts = np.load("./results/NeuralTS_regret.npy")
    # ts_mean, ts_std = get_mean_std(ts)
    # plt.plot(x, ts_mean, 'k-', color='yellow',linewidth=2.0,linestyle=':', label = 'NeuralTS')
    # plt.fill_between(x, ts_mean-ts_std, ts_mean+ts_std, facecolor='green', alpha=0.2)

    # ep = np.load("./results/Neural_epsilon_regret.npy")
    # ep_mean, ep_std = get_mean_std(ep)
    # plt.plot(x, ep_mean, 'k-', color='green',linewidth=2.0,linestyle='-.', label = "NeuralEpsilon")
    # plt.fill_between(x, ep_mean-ep_std, ep_mean+ep_std, facecolor='orange', alpha=0.2)

    plt.xlabel('Rounds')
    plt.ylabel('Regret')
    plt.legend()
    plt.title("Movielens")
    #plt.rcParams["figure.figsize"] = (20, 10)
    plt.savefig('./figures_neurips_two/regret_mnist.pdf', dpi=500)