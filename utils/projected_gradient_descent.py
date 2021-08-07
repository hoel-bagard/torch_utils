# Taken from https://gist.github.com/oscarknagg/45b187c236c6262b1c4bbe2d0920ded6
# Use this one if the first one does not work: https://github.com/Harry24k/PGD-pytorch/blob/master/PGD.ipynb

import torch


# def projected_gradient_descent(model, x, y, loss_fn, num_steps, step_size, step_norm, eps, eps_norm,
#                                clamp=(0, 1), y_target=None):
#     """Performs the projected gradient descent attack on a batch of images."""
#     x_adv = x.clone().detach().requires_grad_(True).to(x.device)
#     targeted = y_target is not None
#     num_channels = x.shape[1]

#     for i in range(num_steps):
#         _x_adv = x_adv.clone().detach().requires_grad_(True)

#         prediction = model(_x_adv)
#         loss = loss_fn(prediction, y_target if targeted else y)
#         loss.backward()

#         with torch.no_grad():
#             # Force the gradient step to be a fixed size in a certain norm
#             if step_norm == 'inf':
#                 gradients = _x_adv.grad.sign() * step_size
#             else:
#                 # Note .view() assumes batched image data as 4D tensor
#                 gradients = _x_adv.grad * step_size / _x_adv.grad.view(_x_adv.shape[0], -1)\
#                     .norm(step_norm, dim=-1)\
#                     .view(-1, num_channels, 1, 1)

#             if targeted:
#                 # Targeted: Gradient descent with on the loss of the (incorrect) target label w.r.t. the image data
#                 x_adv -= gradients
#             else:
#                 # Untargeted: Gradient ascent on the loss of the correct label w.r.t. the model parameters
#                 x_adv += gradients

#         # Project back into l_norm ball and correct range
#         if eps_norm == 'inf':
#             # Workaround as PyTorch doesn't have elementwise clip
#             x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
#         else:
#             delta = x_adv - x

#             # Assume x and x_adv are batched tensors where the first dimension is
#             # a batch dimension
#             mask = delta.view(delta.shape[0], -1).norm(eps_norm, dim=1) <= eps

#             scaling_factor = delta.view(delta.shape[0], -1).norm(eps_norm, dim=1)
#             scaling_factor[mask] = eps

#             # .view() assumes batched images as a 4D Tensor
#             delta *= eps / scaling_factor.view(-1, 1, 1, 1)

#             x_adv = x + delta

#         x_adv = x_adv.clamp(*clamp)

#     return x_adv.detach()


def projected_gradient_descent(model, images, labels, loss, eps=0.3, alpha=2/255, iters=1):
    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels)
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images
