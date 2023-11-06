from model import *

# implementation of "Learning Smooth Neural Functions via Lipschitz Regularization" by Liu et al. 2022
if __name__ == '__main__':
  random.seed(1)

  # hyper parameters
  hyper_params = {
    "dim_in": 2,
    "dim_t": 2,
    "dim_out": 1,
    "h_mlp": [64,64,64,64,64],
    "step_size": 1e-4,
    "grid_size": 32,
    "num_epochs": 2000,
    "samples_per_epoch": 512
  }
  alpha = 1e-6

  # initialize a mlp
  model = lipmlp(hyper_params)
  params = model.initialize_weights()

  # optimizer
  opt_init, opt_update, get_params = optimizers.adam(step_size=hyper_params["step_size"])
  opt_state = opt_init(params)

   # define loss function and update function
  def loss(params_, alpha, x_, y0_, y1_,y2_):
    out0 = model.forward(params_, np.array([0.0,0.0]), x_) # star when t = 0.0
    out1 = model.forward(params_, np.array([1.0,1.0]), x_) # circle when t = 1.0
    out2 = model.forward(params_, np.array([2.0,4.0]), x_) # cross when t = 2.0
    loss_sdf = np.mean(np.sum((out0 - y0_)**2),axis = 1) + np.mean((out1 - y1_)**2) + np.mean(np.sum((out2 - y2_)**2))
    loss_lipschitz = model.get_lipschitz_loss(params_)
    return loss_sdf + alpha * loss_lipschitz

  @jit
  def update(epoch, opt_state, alpha, x_, y0_, y1_,y2_):
    params_ = get_params(opt_state)
    value, grads = value_and_grad(loss, argnums = 0)(params_, alpha, x_, y0_, y1_,y2_)
    opt_state = opt_update(epoch, grads, opt_state)
    return value, opt_state

  # training
  loss_history = onp.zeros(hyper_params["num_epochs"])
  pbar = tqdm.tqdm(range(hyper_params["num_epochs"]))
  for epoch in pbar:
    # sample a bunch of random points
    x = np.array(random.rand(hyper_params["samples_per_epoch"], hyper_params["dim_in"]))
    y0 = jgp.sdf_star(x)
    y1 = jgp.sdf_circle(x)
    y2 = jgp.sdf_cross(x)
    # update
    loss_value, opt_state = update(epoch, opt_state, alpha, x, y0, y1,y2)
    loss_history[epoch] = loss_value
    pbar.set_postfix({"loss": loss_value})

    if epoch % 1000 == 0: # plot loss history every 1000 iter
      plt.close(1)
      plt.figure(1)
      plt.semilogy(loss_history[:epoch])
      plt.title('Reconstruction loss + Lipschitz loss')
      plt.grid()
      plt.savefig("lipschitz_mlp_loss_history.jpg")

  # save final parameters
  params = get_params(opt_state)
  with open("lipschitz_mlp_params.pkl", 'wb') as handle:
    pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

  # normalize weights during test time
  params_final = model.normalize_params(params)

  # save result as a video
  sdf_cm = mpl.colors.LinearSegmentedColormap.from_list('SDF', [(0,'#eff3ff'),(0.5,'#3182bd'),(0.5,'#31a354'),(1,'#e5f5e0')], N=256)

  # create video
  fig = plt.figure()
  x = jgp.sample_2D_grid(hyper_params["grid_size"]) # sample on unit grid for visualization
  def animate(t):
      plt.cla()
      out = model.forward_eval(params_final, np.array([t]), x)
      levels = onp.linspace(-0.5, 0.5, 21)
      im = plt.contourf(out.reshape(hyper_params['grid_size'],hyper_params['grid_size']), levels = levels, cmap=sdf_cm)
      plt.axis('equal')
      plt.axis("off")
      return im

  anim = animation.FuncAnimation(fig, animate, frames = np.linspace(0,2,50), interval=50)
  anim2 = animation.FuncAnimation(fig, animate, frames=np.mgrid[1.0:2.0:50j, 2.0:3.0:50j], interval=50)
  
  anim.save("lipschitz_mlp_interpolation.mp4")
  anim2.save("Lipschitz_mlp_interpolation2.mp4")
  # create video for star to circle
fig_star_circle = plt.figure()
x = jgp.sample_2D_grid(hyper_params["grid_size"]) # sample on unit grid for visualization
def animate_star_circle(t):
    plt.cla()
    out = model.forward_eval(params_final, np.array([t]), x)
    levels = onp.linspace(-0.5, 0.5, 21)
    im = plt.contourf(out.reshape(hyper_params['grid_size'], hyper_params['grid_size']), levels=levels, cmap=sdf_cm)
    plt.axis('equal')
    plt.axis("off")
    return im

anim_star_circle = animation.FuncAnimation(fig_star_circle, animate_star_circle, frames=np.linspace(0,5,50), interval=50)
anim_star_circle.save("star_to_circle.mp4")

# create video for circle to cross
fig_circle_cross = plt.figure()
def animate_circle_cross(t):
    plt.cla()
    out = model.forward_eval(params_final, np.array([t]), x)
    levels = onp.linspace(-0.5, 0.5, 21)
    im = plt.contourf(out.reshape(hyper_params['grid_size'], hyper_params['grid_size']), levels=levels, cmap=sdf_cm)
    plt.axis('equal')
    plt.axis("off")
    return im

anim_circle_cross = animation.FuncAnimation(fig_circle_cross, animate_circle_cross, frames=np.mgrid[1.0:2.0:50j, 2.0:3.0:50j], interval=50)
anim_circle_cross.save("circle_to_cross.mp4")

# create video for cross to circle
fig_cross_circle = plt.figure()
def animate_cross_circle(t):
    plt.cla()
    out = model.forward_eval(params_final, np.array([t,t+1]), x)
    levels = onp.linspace(-0.5, 0.5, 21)
    im = plt.contourf(out.reshape(hyper_params['grid_size'], hyper_params['grid_size']), levels=levels, cmap=sdf_cm)
    plt.axis('equal')
    plt.axis("off")
    return im

anim_cross_circle = animation.FuncAnimation(fig_cross_circle, animate_cross_circle, frames=np.mgrid[2.0:3.0:50j, 1.0:2.0:50j], interval=50)
anim_cross_circle.save("cross_to_circle.mp4")
