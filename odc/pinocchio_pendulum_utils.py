import sys

import pinocchio as pin

import time

import hppfcl as fcl

from imageio import get_writer

import numpy as np

import math

from tqdm import tqdm

from panda3d_viewer import Viewer, ViewerConfig
from pinocchio.visualize.panda3d_visualizer import Panda3dVisualizer
#from xvfbwrapper import Xvfb

def sinwave(T, dt, cut, amp, freq,  wo):
    N = int(T/dt) + cut
    nq = len(freq)
    qref = np.zeros((N, nq))
    vref = np.zeros((N, nq))
    for t in range(N):
        s = t * dt
        for j in range(nq):
            a = amp[j]
            f = freq[j]
            w = wo[j]
            #w = np.pi/2
            #qref[t, j] = a / (2. * np.pi) * np.sin(2. * np.pi * f * s + w)
            qref[t, j] = a * np.sin(2. * np.pi * f * s + w)
            vref[t, j] = a * f * 2 * np.pi * np.cos(2. * np.pi * f * s + w)
    return qref, vref


def generate_ref_traj(T, dt, cut, nq, a, f):
    N = int(T/dt) + cut
    qref = np.zeros((N, nq))
    vref = np.zeros((N, nq))
    for i in range(3):
         wo = np.random.random(nq) * 2 * np.pi
         freq = np.random.random(nq) * f
         #freq = np.array([0, 0.1])
         amp = np.random.random(nq) * 2 * np.pi * a #* np.pi
         #amp = np.array([0, 0.3*2*np.pi])
         q, v = sinwave(T, dt, cut, amp, freq,  wo)
         qref += q
         vref += v
    #qref[:, 0] += np.random.random(nq) * np.pi
    return qref, vref


def generate_pendulum_pinocchio(save_path, q, v, model, geom_model, visual_model, with_cart, dt=0.005, fps=20, T=20, scene_scale=1,
                                window_size=160, damping=0, q_ref=None, v_ref=None, Kp=0, Kv=0, cut=10, ref_traj=None,
                                a_ref=1, f_ref=0.3):
    #print('q: {}'.format(q))
    #vdisplay = Xvfb()
    #vdisplay.start()
    config = ViewerConfig()
    config.show_floor(False)
    config.show_axes(False)
    config.show_grid(False)
    config.set_window_size(window_size, window_size)
    config.set_scene_scale(scene_scale)
    viewer = Viewer(window_type='offscreen', config=config)
    #viewer = Viewer(window_type='onscreen', config=config)
    viewer.reset_camera(look_at=(0, 0, 0), pos=(0, 4, 0))
    viewer.set_background_color(color_rgb=[0.7, 0.8, 0.8])
    viewer.append_group('static')
    viewer.append_plane('static', 'bg_plane', (window_size, window_size), frame=([-1, -2.0, 0.0], [0.7, -0.7, 0.7, 0.0]))
    viewer.set_material('static', 'bg_plane', color_rgba=(0.8, 0.5, 0.2, 1.0))
    viz = Panda3dVisualizer(model, geom_model, visual_model)
    # Initialize the viewer.
    try:
        viz.initViewer(viewer=viewer)
    except ImportError as err:
        print("Error while initializing the viewer. It seems you should install gepetto-viewer")
        print(err)
        sys.exit(0)

    try:
        viz.loadViewerModel("pinocchio")
    except AttributeError as err:
        print("Error while loading the viewer model. It seems you should start gepetto-viewer")
        print(err)
        sys.exit(0)
    #q0 = pin.neutral(model)
    #viz.display(q0)
    #v = np.zeros((model.nv))

    videowriter = get_writer(save_path, fps=fps, quality=5)
    N = math.floor(T / dt)
    dt_vid = 1 / fps
    dt_ratio = dt_vid / dt
    t = 0.
    if ref_traj:
        q_ref, v_ref = generate_ref_traj(T, dt, cut, nq=model.nv, a=a_ref, f=f_ref)
        #print("qref shape: {}".format(q_ref.shape))
        #print('vref shape: {}'.format(v_ref.shape))
        if with_cart:
            #print("q[1]: {}".format(q[1]))
            #print("q_ref[1]: {}".format(q_ref[1]))
            #print("q_ref: {}".format(q_ref))
            #pass
            q[1] = q_ref[0][1]
            v[1] = v_ref[0][1]
        else:
            q = q_ref[0]
            v = v_ref[0]

        #print("qref[:, :10]: {}".format(q_ref[:, :10]))
    else:
        q_ref = np.zeros((N, model.nv))
        v_ref = np.zeros((N, model.nv))
    data_sim = model.createData()
    controls = dict()
    qs = dict()  # Dict that will contain pendulum positions
    q_list = []
    for k in tqdm(range(N)):
        #tau_control = np.zeros((model.nv))
        u = - Kp * (q - q_ref[k]) - Kv * (v - v_ref[k])
        if with_cart:
            u[0] = 0
        #print("u:{}".format(u))
        tau_control = -damping * v + u
        #if with_cart:
         #   tau_control[0] = 0
        #tau_control[0] = 0
        a = pin.aba(model, data_sim, q, v, tau_control)  # Forward dynamics
        v += a * dt
        q = pin.integrate(model, q, v * dt)
        #print("q: {}".format(q))
        #viz.display(q)
        #time.sleep(dt)
        t += dt
        if k % dt_ratio == 0:
            #q_temp = [0.1, 3.14]
            q_list.append(np.array(q))
            #image_rgb = viewer.get_screenshot(requested_format='RGB')
            #videowriter.append_data(image_rgb)
            controls[int(k/dt_ratio)] = u.tolist()
            qs[int(k/dt_ratio)] = q.tolist()
    for q in q_list:
        viz.display(q)
        image_rgb = viewer.get_screenshot(requested_format='RGB')
        videowriter.append_data(image_rgb)
        #time.sleep(dt)
    videowriter.close()
    viewer.stop()
    return controls, qs




def create_model(L, system_position, mass_ratio=1, N=2, with_cart=1, body_mass=1, cart_mass=None, cart_radius=None,
                 cart_length_radius_ratio=None, lower_limit=-2, upper_limit=-1):
    """
    :param L: should be a list of size N. List of the lengths of all bars.
    :param system_position: position of the rotation joint? (position de l'interesection entre premiere barre et support)
    :param mass_ratio: only in the case of 2 pendulums so far
    :param N: number of pendulums
    :param with_cart: 0 or 1.
    :param cart_mass: mass of the cart. Only when with_cart is 1
    :param cart_radius: radius of the cart. Only when with_cart is 1
    """
    #N = 2  # number of pendulums
    model = pin.Model()
    geom_model = pin.GeometryModel()

    parent_id = 0

    if with_cart:
        #cart_radius = 0.1
        cart_length = cart_length_radius_ratio * cart_radius
        cart_mass = cart_mass
        joint_name = "joint_cart"

        geometry_placement = pin.SE3.Identity()
        geometry_placement.rotation = pin.Quaternion(np.array([1., 0., 0.]), np.array([0., 0., 1.])).toRotationMatrix()
        geometry_placement.translation[2] = system_position  # Position verticale du cart
        joint_id = model.addJoint(parent_id, pin.JointModelPX(), pin.SE3.Identity(), joint_name)

        body_inertia = pin.Inertia.FromCylinder(cart_mass, cart_radius, cart_length)
        body_placement = geometry_placement
        model.appendBodyToJoint(joint_id, body_inertia,
                                body_placement)  # We need to rotate the inertia as it is expressed in the LOCAL frame of the geometry

        shape_cart = fcl.Cylinder(cart_radius, cart_length)

        geom_cart = pin.GeometryObject("shape_cart", joint_id, shape_cart, geometry_placement)
        geom_cart.meshColor = np.array([1., 0.1, 0.1, 1.])
        geom_model.addGeometryObject(geom_cart)

        parent_id = joint_id
    else:
        base_placement = pin.SE3.Identity()
        base_placement.translation[2] = system_position
        body_radius = 0.1
        shape0 = fcl.Sphere(body_radius)
        # geom0_obj = pin.GeometryObject("base", 0, shape0, pin.SE3.Identity())
        geom0_obj = pin.GeometryObject("base", 0, shape0, base_placement)
        geom0_obj.meshColor = np.array([1., 0.1, 0.1, 1.])
        geom_model.addGeometryObject(geom0_obj)

    joint_placement = pin.SE3.Identity()
    p = np.array([0, 0, system_position])
    joint_placement.translation = p  # position de la barre
    #joint_placement = pin.SE3.Random()
    #print(joint_placement)
    #body_mass = 1.
    body_mass_2 = mass_ratio * body_mass
    body_masses = [body_mass, body_mass_2]
    if N == 4:
        body_masses = [body_mass, body_mass_2, body_mass_2, body_mass_2]
    body_radius = 0.1  # Rayon de la masse et du support

    base_placement = pin.SE3.Identity()
    base_placement.translation[2] = system_position

    """shape0 = fcl.Sphere(body_radius)
    #geom0_obj = pin.GeometryObject("base", 0, shape0, pin.SE3.Identity())
    geom0_obj = pin.GeometryObject("base", 0, shape0, base_placement)
    geom0_obj.meshColor = np.array([1., 0.1, 0.1, 1.])
    geom_model.addGeometryObject(geom0_obj)"""

    for k in range(N):
        joint_name = "joint_" + str(k + 1)
        joint_id = model.addJoint(parent_id, pin.JointModelRY(), joint_placement, joint_name)

        body_inertia = pin.Inertia.FromSphere(body_masses[k], body_radius)
        body_placement = joint_placement.copy()
        body_placement.translation[2] = L[k]  # Longueur de la barre
        model.appendBodyToJoint(joint_id, body_inertia, body_placement)

        geom1_name = "ball_" + str(k + 1)
        shape1 = fcl.Sphere(body_radius)
        geom1_obj = pin.GeometryObject(geom1_name, joint_id, shape1, body_placement)
        geom1_obj.meshColor = np.ones((4))
        geom_model.addGeometryObject(geom1_obj)

        geom2_name = "bar_" + str(k + 1)
        shape2 = fcl.Cylinder(body_radius / 4., body_placement.translation[2])
        shape2_placement = body_placement.copy()
        shape2_placement.translation[2] /= 2.  # position de la barre

        geom2_obj = pin.GeometryObject(geom2_name, joint_id, shape2, shape2_placement)
        geom2_obj.meshColor = np.array([0., 0., 0., 1.])
        geom_model.addGeometryObject(geom2_obj)

        parent_id = joint_id
        joint_placement = body_placement.copy()

    visual_model = geom_model
    model.lowerPositionLimit.fill(lower_limit * math.pi/2)
    model.upperPositionLimit.fill(upper_limit * math.pi/2)
    #model.lowerPositionLimit.fill(-math.pi)
    #print("model.lowerPositionLimit: {}".format(model.lowerPositionLimit))
    #model.upperPositionLimit.fill(-math.pi/2)
    #print("model.upperPositionLimit: {}".format(model.upperPositionLimit))
    if with_cart:
        #print("Verify lower/upper limits")
        model.lowerPositionLimit[0] = model.upperPositionLimit[0] = 0.
        #model.lowerPositionLimit.fill(0)
        #model.upperPositionLimit.fill(math.pi)
        #model.lowerPositionLimit.fill(0)
        #model.upperPositionLimit.fill(0)
        #model.upperPositionLimit[1] = -math.pi #/ 2
        #model.lowerPositionLimit[1] = math.pi / 2
        #model.lowerPositionLimit[0] = model.upperPositionLimit[0] = 0.
        #print("model.upperPositionLimit: {}".format(model.upperPositionLimit))

    #data_sim = model.createData()
    #import pdb;pdb.set_trace()
    """if with_cart:
        new_gravity = - model.gravity.copy()
        model.gravity = new_gravity"""
    return model, geom_model, visual_model
