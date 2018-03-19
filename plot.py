import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

def plot_reward(rewards,file_name):
	fig,ax = plt.subplots()
	plt.plot(rewards)
	plt.ylabel('cumulative_reward')
	plt.xlabel('time')
	plt.show()
	fig.savefig("./outputs/"+file_name, bbox_inches='tight')
