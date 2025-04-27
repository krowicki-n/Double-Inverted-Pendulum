import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ImageMagickWriter
import do_mpc
from casadi import *

# Define model type: 'continuous' time system
model_type = 'continuous'
model = do_mpc.model.Model(model_type)

"""
Physical constants and parameters

"""
# Mass of the cart (kg)
m0 =  .6 
# Mass of the first rod (kg)
m1 = 0.2  
# Mass of the second rod (kg)
m2 = 0.2  
# Length of the first rod (m)
L1 = .6  
# Length of the second rod (m)
L2 = .6  
# Gravitational acceleration (m/s^2)
g = 9.80665  

# Center of mass of rod 1
l1 = L1/2  
# Center of mass of rod 2
l2 = L2/2 
# Inertia of rod 1
J1 = (m1 * l1**2) / 3  
# Inertia of rod 2
J2 = (m2 * l2**2) / 3  

# Helper constants to simplify equations
h1 = m0 + m1 + m2
h2 = m1*l1 + m2*L1
h3 = m2*l2
h4 = m1*l1**2 + m2*L1**2 + J1
h5 = m2*l2*L1
h6 = m2*l2**2 + J2
h7 = (m1*l1 + m2*L1) * g
h8 = m2*l2*g

# Define state variables
pos = model.set_variable('_x', 'pos')  # cart position
theta = model.set_variable('_x', 'theta', (2,1))  # angles of pendulums
dpos = model.set_variable('_x', 'dpos')  # cart velocity
dtheta = model.set_variable('_x', 'dtheta', (2,1))  # angular velocities

# Define control input
u = model.set_variable('_u', 'force')  # force applied to cart

# Define algebraic variables (accelerations)
ddpos = model.set_variable('_z', 'ddpos')  # cart acceleration
ddtheta = model.set_variable('_z', 'ddtheta', (2,1))  # angular accelerations

# Define differential equations (ODEs)
model.set_rhs('pos', dpos)
model.set_rhs('theta', dtheta)
model.set_rhs('dpos', ddpos)
model.set_rhs('dtheta', ddtheta)

# Define Euler-Lagrange equations (dynamic constraints)
euler_lagrange = vertcat(
    h1*ddpos + h2*ddtheta[0]*cos(theta[0]) + h3*ddtheta[1]*cos(theta[1])
    - (h2*dtheta[0]**2*sin(theta[0]) + h3*dtheta[1]**2*sin(theta[1]) + u),
    h2*cos(theta[0])*ddpos + h4*ddtheta[0] + h5*cos(theta[0]-theta[1])*ddtheta[1]
    - (h7*sin(theta[0]) - h5*dtheta[1]**2*sin(theta[0]-theta[1])),
    h3*cos(theta[1])*ddpos + h5*cos(theta[0]-theta[1])*ddtheta[0] + h6*ddtheta[1]
    - (h5*dtheta[0]**2*sin(theta[0]-theta[1]) + h8*sin(theta[1]))
)

model.set_alg('euler_lagrange', euler_lagrange)
# Kinetic Energy for Cart
E_kin_cart = 1 / 2 * m0 * dpos**2
# Kinetic Enery for Pendulum 1
E_kin_p1 = 1 / 2 * m1 * ((dpos + l1 * dtheta[0] * cos(theta[0]))**2 + 
                            (l1 * dtheta[0] * sin(theta[0]))**2) + 1 / 2 * J1 * dtheta[0]**2
# Kinetic Enery for Pendulum 1
E_kin_p2 = 1 / 2 * m2 * ((dpos + L1 * dtheta[0] * cos(theta[0]) + l2 * dtheta[1] * cos(theta[1]))**2 + 
                            (L1 * dtheta[0] * sin(theta[0]) + l2 * dtheta[1] * sin(theta[1]))**2) + 1 / 2 * J2 * dtheta[0]**2
# Delta KE
E_kin = E_kin_cart + E_kin_p1 + E_kin_p2
# Potential Energy of System
E_pot = m1 * g * l1 * cos(theta[0]) + m2 * g * (L1 * cos(theta[0]) + l2 * cos(theta[1]))

model.set_expression('E_kin', E_kin)
model.set_expression('E_pot', E_pot)
model.setup()
"""
Building the Controller

"""
mpc = do_mpc.controller.MPC(model)
# Define predictions
setup_mpc = {
    'n_horizon': 100,
    'n_robust': 0,
    'open_loop': 0,
    't_step': 0.04,
    'state_discretization': 'collocation',
    'collocation_type': 'radau',
    'collocation_deg': 3,
    'collocation_ni': 1,
    'store_full_solution': True,
    'nlpsol_opts': {'ipopt.linear_solver': 'mumps'}
}
mpc.set_param(**setup_mpc)

# Define the Control Objective | Stabalize upright
mterm = model.aux['E_kin'] - model.aux['E_pot'] # terminal cost
lterm = model.aux['E_kin'] - model.aux['E_pot'] # stage cost
mpc.set_objective(mterm=mterm, lterm=lterm)
# Input force restricted through teh objective
mpc.set_rterm(force=1e-2)
#Constrining the Upper and Lower BOunds
mpc.bounds['lower','_u','force'] = -4
mpc.bounds['upper','_u','force'] = 4
mpc.setup()

"""
Estimator

"""
estimator = do_mpc.estimator.StateFeedback(model)
"""
Simulator

"""
   
simulator = do_mpc.simulator.Simulator(model)
params_simulator = {
    'integration_tool': 'idas',
    'abstol': 1e-6,
    'reltol': 1e-6,
    't_step': 0.04
}
simulator.set_param(**params_simulator)
simulator.setup()

"""
Closed Loop Simulation

"""
simulator.x0['theta'] = 0.99*np.pi

x0 = simulator.x0.cat.full()
simulator.set_initial_guess()
mpc.x0 = x0
estimator.x0 = x0

mpc.set_initial_guess()

"""
Visualization

"""
import matplotlib.pyplot as plt
plt.ion()
from matplotlib import rcParams
rcParams['text.usetex'] = False
rcParams['axes.grid'] = True
rcParams['lines.linewidth'] = 2.0
rcParams['axes.labelsize'] = 'xx-large'
rcParams['xtick.labelsize'] = 'xx-large'
rcParams['ytick.labelsize'] = 'xx-large'

mpc_graphics = do_mpc.graphics.Graphics(mpc.data)

def pendulum_bars(x):
    x = x.flatten()
    # Get the x,y coordinates of the two bars for the given state x.
    line_1_x = np.array([
        x[0],
        x[0]+L1*np.sin(x[1])
    ])

    line_1_y = np.array([
        0,
        L1*np.cos(x[1])
    ])

    line_2_x = np.array([
        line_1_x[1],
        line_1_x[1] + L2*np.sin(x[2])
    ])

    line_2_y = np.array([
        line_1_y[1],
        line_1_y[1] + L2*np.cos(x[2])
    ])

    line_1 = np.stack((line_1_x, line_1_y))
    line_2 = np.stack((line_2_x, line_2_y))

    return line_1, line_2

fig = plt.figure(figsize=(16,9))

ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=3)
ax2 = plt.subplot2grid((3, 2), (0, 1))
ax3 = plt.subplot2grid((3, 2), (1, 1))

ax2.set_ylabel('Angle [rad]')
ax3.set_ylabel('Input force [N]')

for ax in [ax2, ax3]:
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

ax3.set_xlabel('time [s]')

mpc_graphics.add_line(var_type='_x', var_name='theta', axis=ax2)
mpc_graphics.add_line(var_type='_u', var_name='force', axis=ax3)

# Now fix their colors
ax2.get_lines()[0].set_color('black')  # pendulum 1
ax2.get_lines()[1].set_color('red')    # pendulum 2
ax3.get_lines()[0].set_color('green')  # force input

ax1.axhline(0,color='black')

bar1 = ax1.plot([],[], '-o', color='black', linewidth=5, markersize=10)
bar2 = ax1.plot([],[], '-o', color='red', linewidth=5, markersize=10)

ax1.set_xlim(-1.8,1.8)
ax1.set_ylim(-1.2,1.2)
ax1.set_axis_off()

fig.align_ylabels()
fig.tight_layout()

"""
Running Open Loop

"""
u0 = mpc.make_step(x0)
# Visualize Open Loop Prediction
line1, line2 = pendulum_bars(x0)
bar1[0].set_data(line1[0],line1[1])
bar2[0].set_data(line2[0],line2[1])
mpc_graphics.plot_predictions()
mpc_graphics.reset_axes()
fig

"""
Running Closed Loop

"""
# Quickly reset the history of the MPC data object.
mpc.reset_history()

n_steps = 300
for k in range(n_steps):
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)

"""
Final Results

"""
from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter

# The function describing the gif:
x_arr = mpc.data['_x']
def update(t_ind):
    line1, line2 = pendulum_bars(x_arr[t_ind])
    bar1[0].set_data(line1[0],line1[1])
    bar2[0].set_data(line2[0],line2[1])
    mpc_graphics.plot_results(t_ind)
    mpc_graphics.plot_predictions(t_ind)
    mpc_graphics.reset_axes()


anim = FuncAnimation(fig, update, frames=n_steps, repeat=False)
gif_writer = ImageMagickWriter(fps=20)
anim.save('anim_dip.gif', writer=gif_writer)


