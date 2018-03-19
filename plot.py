import matplotlib.pylab as plt

def plot_reward(rewards):
	fig,ax = plt.subplots()
	plt.plot(rewards)
	plt.ylabel('cumulative_reward')
	plt.xlabel('time')
	plt.show()
	fig.savefig("./outputs/reward_plot", bbox_inches='tight')