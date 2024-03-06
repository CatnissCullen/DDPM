import math
# Numerical Operations
import random
import numpy as np
# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from torchvision import datasets
# my_utilities
import my_utilities as my_utils

""" Set paths & device """

data_dir, checkpoints_dir, results_dir = 'data/', 'checkpoints/', 'results/'
device = my_utils.register_device()

""" Set model options """

hp = {
	# ============= data ==============
	'train_batch_size': 64,
	# ============= model ==============
	'opt': 'DDIM',
	'eta': 0,
	'input_size': 32,
	'in_chan': 1,
	'out_chan': 1,
	't_chan': 256,
	'fst_filters': 64,
	'lst_chan': 1024,
	'groups': 32,
	'drop_rate': 0.1,
	'train_T': 700,  # DDIM 700
	'sampling_T': 20,  # DDIM 20
	'beta': 'sin',
	# ============= training ==============
	'init_lr': 6e-5,
	'epoch_num': 10,
	'L2': 0,
	# ============= testing =============
	'samples_num': 20
}

""" Create datasets """

raw_train_set = datasets.MNIST('./data', train=True,
                               transform=
                               transforms.Compose([
	                               transforms.Resize((hp['input_size'], hp['input_size'])),
	                               transforms.ToTensor(),
	                               transforms.Normalize((0.5,), (0.5,))
                               ]),
                               download=False)
train_batches = DataLoader(raw_train_set, shuffle=True, batch_size=hp['train_batch_size'], pin_memory=True)

""" Define weight initialization """


def weights_init(m):
	if isinstance(m, nn.Module):
		for name, param in m.named_parameters():
			if param.requires_grad:
				name = name.replace('.', '_')
				m.register_buffer(f"ema_{name}", param.data.clone())
				if not hasattr(m, 'ema_decay'): m.ema_decay = 0.999


""" Define ResBlock & AttnBlock """


class ResBlock(nn.Module):
	def __init__(self, in_chan, out_chan, t_chan, groups=32, drop_rate=0.1):
		"""
		Residual block
		:param in_chan: number of input channels
		:param out_chan: number of output channels
		:param t_chan: number of time-tensor's channels
		"""
		super().__init__()

		self.block = nn.ModuleDict()
		self.in_chan, self.out_chan, self.t_chan = in_chan, out_chan, t_chan

		self.F1, self.S1, self.P1 = 3, 1, 1
		self.F2, self.S2, self.P2 = 1, 1, 0
		self.groups = groups
		self.drop_rate = drop_rate

		self.set_block()

	# self.apply(weights_init)

	def forward(self, X, t):
		in_X = X
		X = self.block['conv1'](X)
		t = self.block['adjust_t'](t)[:, :, None, None]  # resize to same dims as X
		y = self.block['conv2'](X + t)  # embed
		y += self.block['adjust_x'](in_X)  # shortcut
		return y

	def set_block(self):
		""""""
		""" conv input X """
		self.block['conv1'] = nn.Sequential(
			nn.GroupNorm(self.groups, self.in_chan),
			nn.SiLU(),
			nn.Conv2d(self.in_chan, self.out_chan, self.F1, self.S1, self.P1))

		""" adjust t """
		self.block['adjust_t'] = nn.Sequential(
			nn.SiLU(),
			nn.Linear(self.t_chan, self.out_chan))

		""" conv (X + t) """
		self.block['conv2'] = nn.Sequential(
			nn.GroupNorm(self.groups, self.out_chan),
			nn.SiLU(),
			nn.Dropout(self.drop_rate),
			nn.Conv2d(self.out_chan, self.out_chan, self.F1, self.S1, self.P1))

		""" prepare X for shortcut """
		self.block['adjust_x'] = nn.Conv2d(self.in_chan, self.out_chan, self.F2, self.S2, self.P2) \
			if self.in_chan != self.out_chan else nn.Identity()


class AttnBlock(nn.Module):
	def __init__(self, chan, heads=1, d_k=None, groups=32):
		"""
		Self-attention Block
		:param chan: number of input channels
		:param heads: number of attention heads
		:param d_k: number of each head's dims
		"""
		super().__init__()

		self.chan, self.heads = chan, heads
		self.d_k = self.chan if d_k is None else d_k
		self.groups = groups

		self.prj = nn.Linear(self.chan, self.heads * self.d_k * 3)
		self.softmax = nn.Softmax(dim=2)
		self.mlp = nn.Linear(self.heads * self.d_k, self.chan)

	# self.apply(weights_init)

	def forward(self, X):
		# get q, k, v
		b, ch, h, w = X.shape
		X = X.view(b, ch, -1).permute(0, 2, 1)  # (b, ch, h, w) => (b, h*w, ch) = (b, seq_len, embed_ch)
		q, k, v = torch.chunk(self.prj(X).view(b, -1, self.heads, self.d_k * 3), 3, dim=-1)
		# self-attention
		attn = self.softmax(torch.einsum('bihd,bjhd->bijh', q, k) * (self.d_k ** -0.5))
		scores = torch.einsum('bijh,bjhd->bihd', attn, v).view(b, -1, self.heads * self.d_k)
		y = self.mlp(scores)
		# shortcut
		y = (y + X).permute(0, 2, 1).view(b, ch, h, w)
		return y


""" Define Unet architecture """


class Unet(nn.Module):
	def __init__(self, img_size=32, in_chan=3, out_chan=3, t_chan=256, fst_filters=64, lst_chan=1024, groups=32,
	             drop_rate=0.1, verbose=False):
		"""
		Complete Architecture
		:param img_size: width (or height) of input img.
		:param in_chan: number of input channels
		:param out_chan: number of output channels
		:param t_chan: number of time tensor channels
		:param fst_filters: number of filters in the first conv.
		:param lst_chan: number of channels of the encoder's output
		:param groups: number of groups in GroupNorm
		:param drop_rate: dropout's rate
		:param verbose: whether to print messages in the process
		"""
		super().__init__()
		self.encoder = nn.ModuleList()
		self.bridge = nn.ModuleDict()
		self.decoder = nn.ModuleList()

		self.N = img_size
		self.in_chan, self.out_chan, self.t_chan = in_chan, out_chan, t_chan

		self.fst_filter, self.lst_chan = fst_filters, lst_chan
		self.norm, self.groups = nn.GroupNorm, groups
		self.drop_rate = drop_rate

		self.F1, self.S1, self.P1 = 3, 1, 1
		self.F2, self.S2, self.P2 = 4, 2, 1

		self.ver = verbose

		self.layers_num = 4
		# except the first & the last, each layer has conv. & 2 blocks: Res & Self-Attention
		self.en_results = []  # all 3 parts of results are to be appended

		self.pile_encoder()
		if self.ver: print('-----------------------------------------------------')
		self.set_bridge()
		if self.ver: print('-----------------------------------------------------')
		self.pile_decoder()
		if self.ver: print('=====================================================\n')

		self.apply(weights_init)

	def forward(self, X, t):
		t = self.t_vector2tensor(t)
		latent = self.en_forward(X, t)
		latent = self.bridge_forward(latent, t)
		y = self.de_forward(latent, t)

		if self.ver: print("\nforwarding done.\n"
		                   "=====================================================\n")
		return y

	def t_vector2tensor(self, t: torch.Tensor):
		if self.ver: print("init -> t's dims = " + str(t.shape))
		half_dim = self.t_chan // 8
		emb = math.log(10_000) / (half_dim - 1)
		emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
		t = t.unsqueeze(1) * emb.unsqueeze(0)
		if self.ver: print("encoded -> t's dims = " + str(t.shape))
		t = torch.cat((t.sin(), t.cos()), dim=1)
		if self.ver: print("cat -> t's dims = " + str(t.shape))
		t = nn.Linear(self.t_chan // 4, self.t_chan).to(t.device)(t)
		if self.ver: print("linear1 -> t's dims = " + str(t.shape))
		t = nn.SiLU().to(t.device)(t)
		if self.ver: print("Swish -> t's dims = " + str(t.shape))
		t = nn.Linear(self.t_chan, self.t_chan).to(t.device)(t)
		if self.ver: print("linear2 -> t's dims = " + str(t.shape))

		if self.ver: print("t_vector2tensor done.")
		return t

	def pile_encoder(self):  # downwards
		for i in range(self.layers_num):
			""" CONV """
			if i == 0:
				in_chan = self.in_chan
				out_chan = self.fst_filter
				F, S, P = self.F1, self.S1, self.P1
			else:
				in_chan = self.fst_filter * (2 ** (i - 1))
				out_chan = in_chan
				F, S, P = self.F2, self.S2, self.P2
			conv = nn.Conv2d(in_chan, out_chan, F, S, P)
			self.encoder.append(conv)
			if self.ver: print("(conv)")

			""" RES + ATTN block """
			block = nn.ModuleDict()
			if i == 0:
				in_chan = out_chan = self.fst_filter
			elif i == self.layers_num - 1:
				in_chan, out_chan = self.fst_filter * (2 ** (i - 1)), self.lst_chan
			else:
				in_chan = self.fst_filter * (2 ** (i - 1))
				out_chan = in_chan * 2
			block['res'] = ResBlock(in_chan, out_chan, self.t_chan, hp['groups'], hp['drop_rate'])
			block['attn'] = AttnBlock(out_chan, groups=hp['groups'])
			self.encoder.append(block)
			if self.ver: print("(res -> attn)")

			""" RES + ATTN block """
			block = nn.ModuleDict()
			if i == 0:
				in_chan = out_chan = self.fst_filter
			elif i == self.layers_num - 1:
				in_chan = out_chan = self.lst_chan
			else:
				in_chan = out_chan = self.fst_filter * (2 ** i)
			block['res'] = ResBlock(in_chan, out_chan, self.t_chan, hp['groups'], hp['drop_rate'])
			block['attn'] = AttnBlock(out_chan, groups=hp['groups'])
			self.encoder.append(block)
			if self.ver: print("(res -> attn)")

			if self.ver: print("piling en-layer " + str(i + 1) + " done.")

	def set_bridge(self):
		""""""
		""" RES + ATTN + RES block """
		in_chan = out_chan = self.lst_chan
		self.bridge['res1'] = ResBlock(in_chan, out_chan, self.t_chan, hp['groups'], hp['drop_rate'])
		self.bridge['attn'] = AttnBlock(out_chan, groups=hp['groups'])
		self.bridge['res2'] = ResBlock(in_chan, out_chan, self.t_chan, hp['groups'], hp['drop_rate'])
		if self.ver: print("(res -> attn -> res)")
		if self.ver: print("setting bridge done.")

	def pile_decoder(self):  # upwards
		for i in range(self.layers_num):
			""" RES + ATTN block """
			block = nn.ModuleDict()
			if i == 0:
				in_chan, out_chan = self.lst_chan * 2, self.lst_chan
			else:
				in_chan = self.lst_chan // (2 ** i)
				out_chan = in_chan // 2
			block['res'] = ResBlock(in_chan, out_chan, self.t_chan, hp['groups'], hp['drop_rate'])
			block['attn'] = AttnBlock(out_chan, groups=hp['groups'])
			self.decoder.append(block)
			if self.ver: print("(res -> attn)")

			""" RES + ATTN block """
			block = nn.ModuleDict()
			if i == 0:
				in_chan, out_chan = self.lst_chan * 2, self.lst_chan
			else:
				in_chan = self.lst_chan // (2 ** i)
				out_chan = in_chan // 2
			block['res'] = ResBlock(in_chan, out_chan, self.t_chan, hp['groups'], hp['drop_rate'])
			block['attn'] = AttnBlock(out_chan, groups=hp['groups'])
			self.decoder.append(block)
			if self.ver: print("(res -> attn)")

			""" RES + ATTN block """
			block = nn.ModuleDict()
			if i == 0:
				cat_chan = self.fst_filter * (2 ** (self.layers_num - 2))
				in_chan, out_chan = self.lst_chan + cat_chan, cat_chan
			elif i == self.layers_num - 1:
				cat_chan = self.fst_filter
				in_chan, out_chan = cat_chan + cat_chan, cat_chan
			else:
				cat_chan = self.lst_chan // (2 ** (i + 2))
				in_chan, out_chan = cat_chan * 2 + cat_chan, cat_chan
			block['res'] = ResBlock(in_chan, out_chan, self.t_chan, hp['groups'], hp['drop_rate'])
			block['attn'] = AttnBlock(out_chan, groups=hp['groups'])
			self.decoder.append(block)
			if self.ver: print("(res -> attn)")

			""" CONV^T """
			if i == self.layers_num - 1:
				in_chan = self.fst_filter
				out_chan = self.out_chan
				F, S, P = self.F1, self.S1, self.P1
			else:
				in_chan = self.fst_filter * (2 ** (self.layers_num - i - 2))
				out_chan = in_chan
				F, S, P = self.F2, self.S2, self.P2
			convT = nn.ConvTranspose2d(in_chan, out_chan, F, S, P)
			self.decoder.append(convT)
			if self.ver: print("(convT)")

			if self.ver: print("piling de-layer " + str(i + 1) + " done.")

	def en_forward(self, X, t):
		if self.ver: print('=====================================================')
		for i, layer in enumerate(self.encoder, start=1):
			if not isinstance(layer, nn.ModuleDict):
				self.en_results += [layer(X)]
				if self.ver: print("conv...")
			else:
				self.en_results += [layer['attn'](layer['res'](X, t))]
				if self.ver: print("res... attn...")
			X = self.en_results[-1]
			if self.ver: print("X's size = " + str(X.shape))
			if self.ver: print("forward of en-sublayer " + str(i) + " done.")
			if self.ver: print('--------------------------------------')

		if self.ver: print("en-forward done.")
		return X

	def bridge_forward(self, X, t):
		if self.ver: print('=====================================================')
		X = self.bridge['res2'](self.bridge['attn'](self.bridge['res1'](X, t)), t)
		if self.ver: print("res... attn... res...")
		if self.ver: print("X's size = " + str(X.shape))
		if self.ver: print("bridge-forward done.")
		return X

	def de_forward(self, X, t):  # X is result of bridge
		if self.ver: print('=====================================================')
		for i, layer in enumerate(self.decoder, start=1):
			if not isinstance(layer, nn.ModuleDict):
				X = layer(X)
				if self.ver: print("convT...")
			else:
				cat_X = torch.concat((X, self.en_results.pop()), dim=1)
				X = layer['attn'](layer['res'](cat_X, t))
				if self.ver: print("res... attn...")
			if self.ver: print("X's size = " + str(X.shape))
			if self.ver: print("forward of de-sublayer " + str(i) + " done.")
			if self.ver: print('--------------------------------------')

		if self.ver: print("de-forward done.")
		return X


""" Define EMA update (right after model update) """


def update_ema(m):
	with torch.no_grad():
		for name, param in m.named_parameters():
			if param.requires_grad:
				ema_name = f"ema_{name.replace('.', '_')}"
				ema_param = getattr(m, ema_name)
				ema_param.mul_(model.ema_decay).add_(param.data, alpha=1 - model.ema_decay)


""" Define EMA_params application (before validation) """


def apply_ema(m):
	org_params = {}
	with torch.no_grad():
		for name, param in m.named_parameters():
			if param.requires_grad:
				org_param_name = f"original_{name.replace('.', '_')}"
				org_params[org_param_name] = param.data.clone()
				ema_name = f"ema_{name.replace('.', '_')}"
				ema_param = getattr(m, ema_name)
				param.data.copy_(ema_param)
	return org_params


""" Define params_restoration """


def restore_org_params(m, org_params):
	with torch.no_grad():
		for name, param in m.named_parameters():
			if param.requires_grad:
				org_param_name = f"original_{name.replace('.', '_')}"
				if org_param_name in org_params:
					param.data.copy_(org_params[org_param_name])


""" Diffusion using X0 to get Xt """


def diffuse(X0, beta, t, device, noise=None):
	"""
	Diffusion Process from X0 to Xt
	:param X0: ground-truth image
	:param beta: scheduled variance, larger t larger beta
	:param t: time vector
	:param noise: noise input
	:param device: device
	:return: Xt, noisy img. at t
	"""
	# get mean & var --- q(Xt|X0) = N(Xt; sqrt(alpha_bar_t) * X0, 1-alpha_bar_t)
	alpha_bar_t = my_utils.get_alpha_bar_t(beta, t).to(device)
	mean = torch.sqrt(alpha_bar_t) * X0
	var = 1 - alpha_bar_t
	# diffuse to Xt
	if noise is None: noise = torch.randn_like(X0).to(device)
	Xt = mean + torch.sqrt(var) * noise
	return Xt


def train_DDPM(predictor: Unet, batch, loss_func, optimizer, device, reg_lambda=None, verbose=False):
	"""
	Training process: diffuse -> predict noise with Unet -> get loss
	:param predictor: Unet model for noise predicting
	:param batch: each element is a ground-truth img. => X0
	:param loss_func: loss function (MSE)
	:param optimizer: Adam with default settings
	:param device: device
	:param verbose: whether to print messages in the process
	:param reg_lambda: regularize weight
	:return: noise_loss, the loss between pred_noise & ground-truth diffusion noise
	"""
	# prepare inputs
	batch = batch.to(device)
	batch_size = batch.shape[0]
	t = my_utils.get_t(batch_size, hp['train_T']).to(device)
	beta = my_utils.variance_schedule(hp['train_T'], hp['beta']).to(device)
	noise = torch.randn_like(batch).to(device)
	# diffusion
	Xt = diffuse(batch, beta, t, device, noise)
	# predict noise
	optimizer.zero_grad()
	with autocast():
		predictor.zero_grad()
		pred_noise = predictor(Xt, t)
		if verbose: print("Predicting diffusion noise done.")
		# get loss
		if reg_lambda is not None:
			reg_loss_func = nn.L1Loss()
			reg_loss = reg_lambda * reg_loss_func(pred_noise, noise)
		else:
			reg_loss = 0
		noise_loss = loss_func(pred_noise, noise) + reg_loss
		if verbose: print("Got loss of diffusion noise: ", str(noise_loss.data.item()))
		scaler.scale(noise_loss).backward()
		scaler.step(optimizer)
		scaler.update()
		update_ema(predictor)
		optimizer.zero_grad()
	return noise_loss.data.item()


""" Sampling using trained model (sample after each epoch) """


#  X1 -> X0
def discrete_decoder(X1, beta, device, opt='DDPM'):
	# predict noise_t
	t0 = X1.new_full((X1.shape[0],), 0, dtype=torch.long)
	pred_noise = model.forward(X1, t0)
	alpha_0 = my_utils.get_alpha_t(beta, t0).to(device)
	alpha_bar_0 = my_utils.get_alpha_bar_t(beta, t0).to(device)
	X0 = None
	if opt == 'DDIM':
		pred_X0 = (X1 - torch.sqrt(1 - alpha_bar_0) * pred_noise) / torch.sqrt(alpha_bar_0)  # mean
		# alpha_init = 1  # the paper assume the alpha_0 to be 1, here alpha_init (alpha from 0 to T-1 following t)
		# so sigma_0 = 0, no noise
		X0 = pred_X0
	elif opt == "DDPM":
		alpha_bar_0 = my_utils.get_alpha_bar_t(beta, t0)
		mean = (X1 - ((1 - alpha_0) * pred_noise) / torch.sqrt(1 - alpha_bar_0)) / torch.sqrt(alpha_0)
		var = 1 - alpha_0
		eps = 1e-5
		delta_plus = torch.where(
			X1 >= 1 - eps,
			float('inf'),
			1 / (X1 + 1 + eps) / 255
		)
		delta_minus = torch.where(
			X1 <= -1 + eps,
			float('-inf'),
			1 / (1 - X1 + eps) / 255
		)
		mean = torch.where(
			X1 >= 1 - eps,
			delta_plus,
			torch.where(
				X1 <= -1 + eps,
				delta_minus,
				mean
			)
		)
		prob = torch.distributions.Normal(mean, torch.sqrt(var))  # P(X0|X1)
		X0 = prob.sample()
	return X0


#  others
def denoise(X, beta, t, nxt_t, device, opt='DDPM', eta=0):
	"""
	Denoise process from X{t+1} to X{nxt_t+1}
	:param X: noisy img. at t+1
	:param beta: scheduled variance, larger t larger beta
	:param t: time step vector
	:param nxt_t: next t vector
	:param device: device
	:param opt: DDPM or DDIM
	:param eta: interpolation between DDPM & DDIM
	:return: denoised X
	"""
	# predict noise_t
	pred_noise = model.forward(X, t)
	# get mean & var --- P(X{t-1}|Xt) = N(X{t-1}; mu(Xt, t, pred_noise), beta_t)
	alpha_t = my_utils.get_alpha_t(beta, t).to(device)
	alpha_bar_t = my_utils.get_alpha_bar_t(beta, t)
	noise = torch.randn(X.shape, device=device)
	if opt == 'DDIM':
		pred_X0 = (X - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
		alpha_bar_nxt_t = my_utils.get_alpha_bar_t(beta, nxt_t)
		# beta = 1 - alpha_bar_t / alpha_bar_nxt_t
		# var(DDIM) = eta*(1 - alpha_bar_nxt_t) / (1 - alpha_bar_t)*beta
		var = eta * (1 - alpha_bar_nxt_t) / (1 - alpha_bar_t) * \
		          (1 - alpha_bar_t / alpha_bar_nxt_t)
		dirct2Xt = torch.sqrt(1 - alpha_bar_nxt_t - var) * pred_noise
		X = torch.sqrt(alpha_bar_nxt_t) * pred_X0 + dirct2Xt + torch.sqrt(var) * noise
	elif opt == 'DDPM':
		mean = (X - ((1 - alpha_t) * pred_noise) / torch.sqrt(1 - alpha_bar_t)) / torch.sqrt(alpha_t)
		var = 1 - alpha_t
		X = mean + torch.sqrt(var) * noise
	return X


"""
	Complete Training
"""

model = Unet(
	hp['input_size'],
	hp['in_chan'],
	hp['out_chan'],
	hp['t_chan'],
	hp['fst_filters'],
	hp['lst_chan'],
	hp['groups'],
	hp['drop_rate']
).to(device)

loss_func = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=hp['init_lr'], weight_decay=hp['L2'])

# set epoch_num
e_num = hp['epoch_num']

# train loop
e_loss = []  # losses of each epoch
min_loss = None  # save the minimum loss (after each epoch)
for e in range(e_num):  # iter. epochs
	""" training """
	model.train()
	b_loss, b, tol_loss = [], 0, 0  # b_loss: losses (average) of each batch
	for batch in train_batches:  # iter. batches
		b += 1
		loss = train_DDPM(model, batch[0], loss_func, optimizer, device)
		b_loss.append(loss)
		tol_loss += loss
		if (b - 1) % 100 == 0: print("epoch ", e + 1, "/", e_num, "train_batch ", b, " => loss: ", loss)
	batch_num = b
	avg_loss = tol_loss / batch_num
	print("epoch ", e + 1, "/", e_num, " => avg. train_loss: ", avg_loss)
	org_params = apply_ema(model)  # save original params
	if min_loss is None or avg_loss < min_loss:
		min_loss = avg_loss
		my_utils.save_model_chk_point(
			checkpoints_dir,
			e, model, min_loss,
			optimizer
		)

	""" sampling """
	model.eval()
	with torch.no_grad():
		Xt = torch.randn([hp['samples_num'], hp['in_chan'], hp['input_size'], hp['input_size']], device=device)
		beta = my_utils.variance_schedule(hp['train_T'], hp['beta']).to(device)
		c = hp['train_T'] // hp['sampling_T']
		t_seq = list(range(0, hp['train_T'], c))
		for i in range(len(t_seq)-1, -1, -1):  # t from T-1 to 0
			if i != 0:
				t, nxt_t = t_seq[i], t_seq[i-1]
				t = Xt.new_full((hp['samples_num'],), t, dtype=torch.long)
				nxt_t = Xt.new_full((hp['samples_num'],), nxt_t, dtype=torch.long)
				Xt = denoise(Xt, beta, t, nxt_t, device, hp['opt'], hp['eta'])
			else:
				X0 = discrete_decoder(Xt, beta, device, hp['opt'])
		my_utils.save_gen_chk_point(X0, results_dir, e + 1)

	""" restore org_params """
	restore_org_params(model, org_params)
