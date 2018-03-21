import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

def plot_reward(rewards,file_name):
	fig,ax = plt.subplots()
	plt.plot(rewards)
	plt.ylabel('cumulative_reward')
	plt.xlabel('time')
	#plt.show()
	fig.savefig("./outputs/"+file_name, bbox_inches='tight')

def plot_reward_all(curves,curve_names,file_name):
	m = ['s','.','o','*','+','s','_','*']
	colors=['b', 'g', 'r', 'm', 'k', 'y', 'k', 'w']
	fig,ax = plt.subplots()
	for i in range(len(curves)):
		plt.plot(curves[i][0::5],color=colors[i],marker=m[i], label=curve_names[i],markevery=5+(2*i),markersize=10,linewidth=1.5)
	        #plt.plot(curves[i][0::20,1], curves[i][0::20,0],color=colors[i],marker=m[i], label=curve_names[i].split("/")[1].split(".")[0])

	plt.tick_params(axis='both', which='major', labelsize=18)
	plt.tick_params(axis='both', which='minor', labelsize=18)
	#plt.rcParams.update(params)
	plt.legend(prop={'size':22},loc=1)
	#plt.axis([0, 0.4, 0, 1.0])

	plt.xlabel('steps',fontsize=22)
	plt.ylabel('reward',fontsize=22)
	plt.show()
	fig.savefig("./outputs/"+file_name, bbox_inches='tight')