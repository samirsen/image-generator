import argparse
import matplotlib.pyplot as plt
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--losses')
parser.add_argument('-train', action='store_true')
parser.add_argument('-validation', action='store_true')

args = parser.parse_args()

def plot_losses(mode):
	losses = torch.load(args.losses)
	gen_losses = losses[mode]['generator']
	dis_losses = losses[mode]['discriminator']

	num_epochs = gen_losses[-1][1]

	gen_losses = smooth_losses(gen_losses)
	dis_losses = smooth_losses(dis_losses)

	assert(num_epochs == len(gen_losses))
	
	print("Plotting losses for %d epochs" % num_epochs)

	plt.plot(range(num_epochs), gen_losses)
	plt.plot(range(num_epochs), dis_losses)
	plt.show()

def smooth_losses(losses):
	new_losses = []
	sum_loss = 0.0
	count = 0
	current_epoch = 0
	for loss_tuple in losses:
		epoch = loss_tuple[1]
		if epoch == current_epoch:
			sum_loss += loss_tuple[0]
			count += 1
		else:
			new_losses.append(sum_loss / count)
			current_epoch = epoch
			sum_loss = loss_tuple[0]
			count = 1
	return new_losses

if __name__ == '__main__':
	if args.losses:
		if args.train and not args.validation:
			plot_losses('train')
		elif args.validation and not args.train:
			plot_losses('val')
		else:
			print("Need to know what to plot")
	else:
		print("Invalid arguments")

