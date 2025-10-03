import numpy as np
import pylab as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class SuperLoss(nn.Module):
    """ Super Loss =
        (loss_func(x) - expectation(loss)) * conf

    The confidence is deterministically computed as
        conf = argmin_c [(loss_func(x) - expecation) * c + weight_decay * log(c)**2]
        i.e., as the fixpoint of the confidence in a dynamic system.

    ncls:       number of classes
    nsamples:   number of training instances

    wd_cls:     weight decay for the classes
    wd_ins:     weight decay for the instances

    smooth_cls: smoothing parameter for the class losses
    smooth_ins: smoothing parameter for the instance losses

    loss_func:  loss function that will be used.
    store_conf: if True, we store the instance confidence in the model at every epoch
    """

    def __init__(self, nsamples, ncls, wd_cls=0, wd_ins=0, expectation=0,
                 smooth_cls=0, smooth_ins=0, smooth_init=0, mode='metaloss',
                 loss_func=nn.CrossEntropyLoss(), store_conf=False):
        super().__init__()
        assert ncls > 0 and nsamples > 0, 'need to know the number of class and labels'
        self.ncls = ncls
        self.class_smoother = Smoother(smooth_cls, ncls, init=smooth_init)
        self.optimal_conf_cls = make_optimal_conf(wd_cls, mode)

        self.nsamples = nsamples
        self.instance_smoother = Smoother(smooth_ins, nsamples, init=smooth_init)
        self.optimal_conf_ins = make_optimal_conf(wd_ins, mode)

        self.loss_func = loss_func
        assert hasattr(self.loss_func, 'reduction')
        self.loss_func.reduction = 'none'
        self.expectation = make_expectator(expectation)
        self.store_conf = store_conf

    def forward(self, preds, labels, indices, **kw):
        # compute loss for each sample
        loss_batch = self.loss_func(preds, labels)

        conf_ins = conf_cls = 1

        if self.optimal_conf_ins:
            smoothed_loss_ins = self.instance_smoother(loss_batch.detach(), indices)
            threshold_ins = self.expectation(smoothed_loss_ins)
            conf_ins = self.optimal_conf_ins(smoothed_loss_ins - threshold_ins)
            self.expectation.update(smoothed_loss_ins, conf_ins)

        if self.optimal_conf_cls:
            smoothed_loss_cls = self.class_smoother(loss_batch.detach(), labels)
            threshold_cls = self.expectation(smoothed_loss_cls)
            conf_cls = self.optimal_conf_cls(smoothed_loss_cls - threshold_cls)
            self.expectation.update(smoothed_loss_cls, conf_cls)

        conf = conf_ins * conf_cls

        # try:
        #     print(f'index of minimum loss: {loss_batch.tolist()[0].index(min(loss_batch.tolist()[0]))}, minimum loss value: {min(loss_batch.tolist()[0])}')
        #     print(f'index of maximum loss: {loss_batch.tolist()[0].index(max(loss_batch.tolist()[0]))}, maximum loss value: {max(loss_batch.tolist()[0])}')
        #     threshold = loss_batch.mean().item()
        #     loss = loss_batch * (loss_batch <= threshold)
        #     # loss = loss_batch * (loss_batch >= threshold)
        #     loss = loss.mean()
        # except:
        #     # compute the final loss
        #     loss = (loss_batch * conf).mean()
        loss = (loss_batch * conf).mean()
        return loss


class Smoother(nn.Module):
    def __init__(self, smoothing, nsamples, init=0):
        super().__init__()
        assert 0 <= smoothing < 1
        self.smoothing = smoothing
        self.nsamples = nsamples
        if self.smoothing:
            if isinstance(init, (int, float)):
                assert nsamples > 0
                init = torch.full([nsamples], init)
            self.register_buffer('memory', init.clone())

    def __call__(self, values, indices=None):
        if self.smoothing > 0:
            assert len(values) == len(indices)
            binned_values = torch.bincount(indices, weights=values, minlength=self.nsamples)
            bin_size = torch.bincount(indices, minlength=self.nsamples).float()
            nnz = (bin_size > 0)  # which classes are represented
            means = binned_values[nnz] / bin_size[nnz]  # means for each class
            alpha = self.smoothing ** bin_size[nnz]
            self.memory[nnz] = alpha * self.memory[nnz] + (1 - alpha) * means  # update
            return self.memory[indices]
        else:
            return values


def precomp_x_equal_exp_x(method='newton'):
    """ precompute the solution of
       min (a/e^-x + x^2)
     equivalent to
       a * e^x + 2x == 0
    Here, x is the log(confidence). This is linked to the W function, see
    https://fr.wikipedia.org/wiki/Fonction_W_de_Lambert
    Example:
      * if loss is positive == 1 (i.e. bad sample)
        loss = 1 / exp(-x)                 ==> optimal conf = -inf
        loss = 1 / exp(-x) + ||conf**2||   ==> optimal conf = -0.35173
      * if loss is negative == -0.5 (i.e. good sample)
        loss = -0.5 / exp(-conf)                 ==> optimal conf = +inf
        loss = -0.5 / exp(-conf) + ||conf**2||   ==> optimal conf = 0.35740
      * if loss < -0.7357, there are no solutions other than conf = +inf
    """
    loss = lambda x, a: a * np.exp(x) + x ** 2
    der1 = lambda x, a: a * np.exp(x) + 2 * x
    der2 = lambda x, a: a * np.exp(x) + 2

    from scipy.optimize import root_scalar
    aa = -0.750256 + np.geomspace(0.01, 1000, 32)
    xs = []
    for a in aa:
        x = root_scalar(der1, (a,), method, fprime=der2, x0=0)
        xs.append(x.root if x.converged else float('nan'))
    xs = np.float32(xs)
    print(aa.tolist())
    print(xs.tolist())

    pl.subplot(211)
    x = np.linspace(-3, 3, 100)
    for a in [-0.5, 0, 1]:
        pl.plot(x, loss(x, a), label=f"a={a}")
        x0 = np.interp(a, aa, xs)
        pl.plot(x0, loss(x0, a), '+')
    pl.legend()
    pl.subplot(212)
    pl.plot(aa, xs)


# coming from precomp_x_equal_exp_x
loss_div_wd = np.float32(
    [-1000, -0.7357585932962737, -0.7292385198866751, -0.7197861042909649,
     -0.7060825529685993, -0.6862159572880272, -0.6574145455480526, -0.6156599675844636,
     -0.5551266577364037, -0.46736905653740307, -0.34014329294487, -0.15569892914556094,
     0.11169756647530316, 0.4993531412919867, 1.0613531942004133, 1.8761075276533326,
     3.0572900212223724, 4.769698321281568, 7.252246278161051, 10.851297017399714,
     16.06898724880869, 23.63328498268829, 34.599555050301056, 50.497802769609315,
     73.54613907594951, 106.96024960367691, 155.40204460004963, 225.63008495214464,
     327.4425312511471, 475.0441754009414, 689.0282819387658, 999.249744])

conf = np.float32(
    [1, 0.9991138577461243, 0.8724386692047119, 0.8048540353775024, 0.7398145198822021,
     0.6715637445449829, 0.5973713397979736, 0.5154045820236206, 0.42423248291015625,
     0.3226756751537323, 0.20976418256759644, 0.08473344892263412, -0.05296758562326431,
     -0.2036692053079605, -0.3674810528755188, -0.5443023443222046, -0.7338425517082214,
     -0.9356498718261719, -1.149145483970642, -1.3736592531204224, -1.6084641218185425,
     -1.8528070449829102, -2.1059343814849854, -2.367111921310425, -2.6356399059295654,
     -2.910861015319824, -3.1921679973602295, -3.479003667831421, -3.770861864089966,
     -4.067285060882568, -4.367861747741699, -4.67222261428833])


def get_optimal_conf(loss, weight_decay):
    assert weight_decay > 0
    return np.interp(loss / weight_decay, loss_div_wd, conf)


class OptimalConf(nn.Module):
    """ Pytorch implementation of the get_optimal_conf() function above
    """

    def __init__(self, weight_decay=1, mode='torch'):
        super().__init__()
        self.weight_decay = weight_decay
        self.mode = mode
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # transformation from: loss_div_wd[1:] --> [0, ..., len(loss_div_wd)-2]
        log_loss_on_wd = torch.log(torch.from_numpy(loss_div_wd[1:]) + 0.750256)
        step = (log_loss_on_wd[-1] - log_loss_on_wd[0]) / (len(log_loss_on_wd) - 1)
        offset = log_loss_on_wd[0]

        # now compute step and offset such that [0,30] --> [-1,1]
        self.log_step = step * (len(log_loss_on_wd) - 1) / 2
        self.log_offset = offset + self.log_step
        self.register_buffer('optimal_conf', torch.from_numpy(conf[1:]).view(1, 1, 1, -1))

    def __call__(self, loss):
        loss = loss.detach()
        if self.mode == 'numpy':
            conf = get_optimal_conf(loss.cpu().numpy(), self.weight_decay)
            r = torch.from_numpy(conf).to(loss.device)

        elif self.mode == 'torch':
            l = loss / self.weight_decay
            l = 1 - l
            # using grid_sampler in the log-space of loss/wd
            l = (torch.log(l + 0.750256) - self.log_offset) / self.log_step
            l[torch.isnan(l)] = -1  # not defined before -0.75
            l = torch.stack((l, l.new_zeros(l.shape)), dim=-1).view(1, 1, -1, 2)
            self.optimal_conf = self.optimal_conf.to(self.device)
            r = F.grid_sample(self.optimal_conf, l, padding_mode="border", align_corners=True) # first input is INPUT, second input is GRID
        return torch.exp(r.view(loss.shape))


def make_optimal_conf(wd, mode):
    if wd == 0:
        return None
    elif mode == 'metaloss':
        return OptimalConf(wd)
    else:
        raise ValueError('bad mode ' + mode)


class Constant(nn.Module):
    def __init__(self, expectation):
        super().__init__()
        self.expectation = expectation

    def __call__(self, values):
        return self.expectation

    def update(self, values, weights=None):
        pass


class GlobalAverage(nn.Module):
    def __init__(self, weighted=False):
        super().__init__()
        self.weighted = weighted
        self.register_buffer('sum_values', torch.tensor(0.0))
        self.register_buffer('sum_weights', torch.tensor(0.0))

    def __call__(self, values):
        if self.sum_weights == 0:
            return values.mean()  # special case, only happen once at begining
        else:
            return self.sum_values / self.sum_weights

    def update(self, values, weights=None):
        self.sum_values += (values * weights if self.weighted else values).sum()
        self.sum_weights += weights.sum() if self.weighted else len(values)


class WindowAverage(GlobalAverage):
    def __init__(self, window, weighted=False):
        super().__init__(weighted=weighted)
        self.window = window
        self.list_values = []
        self.list_weights = []

    def sum_pop_list(self, lis):
        res = sum(lis[:-self.window])
        del lis[:-self.window]
        return res

    def update(self, values, weights=None):
        super().update(values, weights)
        self.list_values += (values * weights if self.weighted else values).cpu().numpy().tolist()
        self.list_weights += weights.cpu().numpy().tolist() if self.weighted else [1] * len(values)
        self.sum_values -= self.sum_pop_list(self.list_values)
        self.sum_weights -= self.sum_pop_list(self.list_weights)


class ExpAverage(nn.Module):
    def __init__(self, smooth, weighted=False):
        super().__init__()
        self.weighted = weighted
        assert 0 <= smooth < 1
        self.smooth = smooth
        self.register_buffer('average', torch.tensor(0.0))
        self.register_buffer('weight', torch.tensor(0.0))

    def __call__(self, values):
        if self.average == 0:
            return values.mean()
        else:
            return self.average / self.weight

    def update(self, values, weights=None):
        avg = (values * weights if self.weighted else values).mean()
        w = weights.mean() if self.weighted else 1
        smooth = 0 if self.average == 0 else self.smooth
        self.average.set_(smooth * self.average + (1 - smooth) * avg)
        self.weight.set_(smooth * self.weight + (1 - smooth) * w)


def make_expectator(expectation):
    if expectation is None: return None
    if isinstance(expectation, str):
        expectation = eval(expectation)
    if isinstance(expectation, (int, float)):
        expectation = Constant(expectation)
    return expectation
