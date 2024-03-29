
# Расчет вероятности успешного наведения
from multiprocessing import parent_process
from sqlite3 import connect
from turtle import position
from vispy import app, scene, visuals
from vispy.visuals.transforms import STTransform, MatrixTransform

import threading
import time

import numpy as np
import matplotlib.pyplot as plt 
from numpy import linalg as LA
import math
import random


# учитывая ограничения ЗВНУ ДН


# никак не учитывается, мы на плоскости
y_min = 0
y_max = 10 



# Хлам не нужный, у нас нет пилота
# T_l = 
# tau_l = 



# params
scale_factor = 0.0001
# params radar
detection_range = 80000.0 * scale_factor # дальность обнаружения
total_lead_angle = math.radians(11) 
# opening_angle = math.radians(15)
opening_angle = math.radians(45) # полураствор сектора обнаружения цели
speed_aircraft = 300 * scale_factor
speed_aircraft_target = -300 * scale_factor


delta_D = random.uniform(0, 100) * scale_factor
delta_epsilon = random.uniform(0, math.radians(1))
delta_q = random.uniform(0, math.radians(1))
delta_V = random.uniform(0, 10) * scale_factor
delta_V_target = random.uniform(0, 10) * scale_factor
delta_pilot = random.uniform(0,  math.radians(1)) # дискретный случайный процесс с периодом T_об появления
                                                  # новых случайных значений ошибок штурмана наведения 

# errors(взяты по приколу)
def update_errors():
    global delta_D
    global delta_epsilon
    global delta_q
    global delta_V
    global delta_V_target
    global delta_pilot
    
    delta_D = random.uniform(0, 100) * scale_factor
    delta_epsilon = random.uniform(0, math.radians(1))
    delta_q = random.uniform(0, math.radians(1))
    delta_V = random.uniform(0, 10) * scale_factor
    delta_V_target = random.uniform(0, 10) * scale_factor
    delta_pilot = random.uniform(0, 20) #

def reset_errors():
    global delta_D
    global delta_epsilon
    global delta_q
    global delta_V
    global delta_V_target
    global delta_pilot
    
    delta_D = 0
    delta_epsilon = 0
    delta_q = 0
    delta_V = 0
    delta_V_target = 0
    delta_pilot = 0



# -------------------Graphics----------------------

drawTrajLine = True

Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
Sphere3D = scene.visuals.create_visual_node(visuals.EllipseVisual)

class Visualizer():
    def __init__(self):
        self.canvas3d = scene.SceneCanvas(keys='interactive', bgcolor='grey')
        self.canvasXY = scene.SceneCanvas(keys='interactive', bgcolor='grey')
        self.view = self.canvasXY.central_widget.add_view()
        self.view = self.canvas3d.central_widget.add_view()
        self.view.camera = "turntable"

        xax = scene.Axis(pos=[[0, 0], [5, 0]], tick_direction=(0, -1),
                         axis_color='r', tick_color='r', text_color='r',
                         font_size=16, parent=self.view.scene, domain=(0., 5.))

        yax = scene.Axis(pos=[[0, 0], [0, 5]], tick_direction=(-1, 0),
                         axis_color='g', tick_color='g', text_color='g',
                         font_size=16, parent=self.view.scene, domain=(0., 5.))

        zax = scene.Axis(pos=[[0, 0], [-5, 0]], tick_direction=(0, -1),
                         axis_color='b', tick_color='b', text_color='b',
                         font_size=16, parent=self.view.scene, domain=(0., 5.))
        # # its actually an inverted x-axis
        zax.transform = scene.transforms.MatrixTransform()
        zax.transform.rotate(90, (0, 1, 0))  # rotate cw around y-axis
        zax.transform.rotate(-45, (0, 0, 1))  # tick direction towards (-1,-1)


        self.dataX = [0]
        self.dataY = [0]
        self.dataZ = [0]

        self.sc3d = Scatter3D(parent=self.view.scene)
        self.sc3d_1 = Scatter3D(parent=self.view.scene)
        self.sc3d.set_gl_state('translucent', blend=True, depth_test=True)


        self.range = scene.visuals.Ellipse(center = [0, 0, 0], color = [0, 1, 1, 0.1], radius = [detection_range, detection_range], border_color = [0, 1, 1, 1])
        self.line_trajectory = scene.visuals.Line(pos=[[0, 0], [0, 5]])
        self.view.add(self.line_trajectory)
        
        self.direction_opening = scene.visuals.Axis(pos=[[0, 0], [0, detection_range]], tick_direction=(-1, 0),
                         axis_color=[0.7, 1, 1], tick_color='g', text_color='g',
                         font_size=16, parent=self.view.scene, domain=(0., 5.))
        self.direction = scene.visuals.Axis(pos=[[0, 0], [0, 5]], tick_direction=(-1, 0),
                         axis_color=[0.7, 1, 1, 0.1], tick_color='g', text_color='g',
                         font_size=16, parent=self.view.scene, domain=(0., 5.))
        self.left_direction = scene.visuals.Axis(pos=[[0, 0], [0, detection_range]], tick_direction=(-1, 0),
                         axis_color='y', tick_color='g', text_color='g',
                         font_size=16, parent=self.view.scene, domain=(0., 5.))
        self.right_direction = scene.visuals.Axis(pos=[[0, 0], [0, detection_range]], tick_direction=(-1, 0),
                         axis_color='y', tick_color='g', text_color='g',
                         font_size=16, parent=self.view.scene, domain=(0., 5.))
        self.view.add(self.range)

        self.orientation = 0
        self.group_colors = np.ones((1, 4), dtype=np.float32)
        self.group_poses = np.zeros((1, 3), dtype=np.float32)
        self.update_data = False

    def startVisualization(self):
        # добавляем таймер и запускаем приложение
        timer = app.Timer()
        timer.connect(self._updatePlots)
        timer.start()
        self.canvas3d.show()
        app.run()
    

    def setVehicleStates(self, pose_list, color_list):
        self.group_poses = pose_list
        self.group_colors = color_list
        self.update_data = True
        if drawTrajLine:
            self.dataX.append(self.group_poses[0][0])
            self.dataY.append(self.group_poses[0][1])
            self.dataZ.append(self.group_poses[0][2])

    def get_orientation(self, orientation):
        self.orientation = orientation

    def _updatePlots(self, ev):
        if(self.update_data == True):
            self.sc3d.set_data(self.group_poses, face_color=self.group_colors, symbol='o', size=10,
                edge_width=0.5, edge_color='blue')

            if drawTrajLine:
                dataPlot = np.array([self.dataX, self.dataY, self.dataZ]).T
                self.line_trajectory.set_data(pos=dataPlot, color=(0, 1, 1, 1))

            tr_range = MatrixTransform()
            tr_range.translate((self.group_poses[0][0],
                          self.group_poses[0][1],
                          self.group_poses[0][2]))
            self.range.transform = tr_range

            tr_opening = MatrixTransform()
            tr_direction = MatrixTransform()
            tr_left_direction = MatrixTransform()
            tr_right_direction = MatrixTransform()

            tr_opening.rotate(math.degrees(self.orientation + total_lead_angle), (0, 0, 1))
            tr_direction.rotate(math.degrees(self.orientation), (0, 0, 1))
            tr_left_direction.rotate(math.degrees(self.orientation + total_lead_angle + opening_angle), (0, 0, 1))
            tr_right_direction.rotate(math.degrees(self.orientation + total_lead_angle - opening_angle), (0, 0, 1))

            tr_opening.translate((self.group_poses[0][0],
                                  self.group_poses[0][1],
                                  self.group_poses[0][2]))

            tr_direction.translate((self.group_poses[0][0],
                                    self.group_poses[0][1],
                                    self.group_poses[0][2]))

            tr_left_direction.translate((self.group_poses[0][0],
                                         self.group_poses[0][1],
                                         self.group_poses[0][2]))

            tr_right_direction.translate((self.group_poses[0][0],
                                          self.group_poses[0][1],
                                          self.group_poses[0][2]))

            self.direction.transform = tr_direction
            self.direction_opening.transform = tr_opening
            self.left_direction.transform = tr_left_direction
            self.right_direction.transform = tr_right_direction

            self.update_data = False

vis = Visualizer()

# -------------------------------------------------

class Math:
    def calculate_angle(self, x, y):
        ret_angle = np.arctan2(y , x) + math.radians(180)

        return (ret_angle)

    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::

                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
    def get_angle(self, center, point):
        x = point[0] - center[0]
        y = point[1] - center[1]
        global delta_pilot
        delta_pilot = random.uniform(0,  math.radians(1))

        if (x == 0):
            if (y > 0):
                return(math.radians(180))
            else:
                return (0)
        a = math.atan(y / x) + delta_pilot
        if (x > 0):
            return (a + math.pi / 2)
        else:
            return (a + math.radians(270))



class Math_model_aircraft:
    def __init__(self, velocity, position, phi):

        self.velocity = velocity
        self.position = position
        self.past_velocity = np.array([0, 0, 0])
        self.phi = phi

        self.mathem = Math()

    def rotate_matrix(self, phi):
        array_matrix_2d = np.array([[np.cos(phi), +np.sin(phi)],
                                [-np.sin(phi), np.cos(phi)]])

        return array_matrix_2d

    def calculate_position(self, abs_velocity, phi, dt):
        self.calculate_velocity(abs_velocity, phi)
        self.calculate_angle(phi)
        self.position += self.velocity * dt
        self.past_velocity = self.velocity

    def calculate_velocity(self, abs_velocity, phi):
        self.velocity[0] = abs_velocity * np.cos(self.pid(phi))
        self.velocity[1] = abs_velocity * np.sin(self.pid(phi))
        self.velocity[2] = 0

    def pid(self, desired_angle):
        # error_angle = desired_angle - self.phi
        # k_p = 1
        # cmd = k_p * error_angle

        return(desired_angle)

    def get_velocity(self):
        return self.velocity

    def get_position(self):
        return self.position
        
    def get_phi(self):
        return self.phi

    def calculate_angle(self, phi):
        if (np.linalg.norm(self.velocity) == 0 or np.linalg.norm(self.past_velocity) == 0):
            self.phi += 0
        else:
            self.phi += self.mathem.angle_between(self.velocity, self.past_velocity)

        if (self.phi > 360):
            self.phi = 0
        elif (self.phi < 0):
            self.phi = 360
    
    def update_errors():
        a = 2

class Simulator:
    def __init__(self):

        self.mathem = Math()

        # params aircraft
        self.velocity_airplane = np.array([0, 0, 0]) * scale_factor
        self.position_airplane = np.array([0.0, 0.0, 0]) * scale_factor
        self.phi = 0

        # params aircraft target
        # я взял диапазон генерации чисел случайный, в лабе ни слова про радиус действия РЛС
        x_target_init = random.uniform(-75000.0, 75000)
        y_target_init = random.uniform(-250000.0, 250000.0)
        self.position_target = np.array([x_target_init, y_target_init, 0]) * scale_factor
        self.velocity_target = np.array([0.0, 0.0, 0]) * scale_factor

        # params simulator
        self.dt = 0.1
        self.aircraft = Math_model_aircraft(self.velocity_airplane, self.position_airplane, self.phi)
        self.aircraft_target = Math_model_aircraft(self.velocity_target, self.position_target, math.radians(90))
        
    def update_init_param(self):
        # params aircraft
        self.velocity_airplane = np.array([0, 0, 0]) * scale_factor
        self.position_airplane = np.array([0.0, 0.0, 0]) * scale_factor
        self.phi = 0

        # params aircraft target
        # я взял диапазон генерации чисел случайный, в лабе ни слова про радиус действия РЛС
        x_target_init = random.uniform(-75000.0, 75000)
        y_target_init = random.uniform(-250000.0, 250000.0)
        self.position_target = np.array([x_target_init, y_target_init, 0]) * scale_factor
        self.velocity_target = np.array([0.0, 0.0, 0]) * scale_factor

        # params simulator
        self.dt = 0.1
        self.aircraft = Math_model_aircraft(self.velocity_airplane, self.position_airplane, self.phi)
        self.aircraft_target = Math_model_aircraft(self.velocity_target, self.position_target, math.radians(90))


    def dotproduct(self, v1, v2):
        return sum((a*b) for a, b in zip(v1, v2))

    def length(self, v):
        return math.sqrt(self.dotproduct(v, v))

    def angle(self, v1, v2):
        return math.acos(self.dotproduct(v1, v2) / (self.length(v1) * self.length(v2)))

    def rotate_matrix(self, phi):
        array_matrix_2d = np.array([[np.cos(phi),   -np.sin(phi),   0],
                                    [+np.sin(phi),  np.cos(phi),    0],
                                    [0,             0,              1]])
        
        return array_matrix_2d


    def is_detection(self, position_aircraft_target, position_aircraft, phi):
        distance = np.linalg.norm(np.array([0.0, detection_range]))
        between_distance = np.linalg.norm(position_aircraft - position_aircraft_target) + delta_D

        vec = self.aircraft.get_velocity() * 1000 + delta_V

        right_line = self.rotate_matrix(opening_angle) @ vec
        left_line = self.rotate_matrix(-opening_angle) @ vec

        distance_point_right = np.cross(position_aircraft_target - position_aircraft, position_aircraft - right_line)[2]
        distance_point_left = np.cross(position_aircraft_target - position_aircraft, position_aircraft - left_line)[2]

        if (between_distance <= distance):
            if (distance_point_right <= 0 and distance_point_left >= 0):
                return True
        else:
            return False

    def is_continue(self, t, phi):
        T_d = 30 # время полёта истребителя на полный радиус действия(каво?)
        position = self.aircraft.get_position()
        position_target = self.aircraft_target.get_position()
        velocity = self.aircraft.get_velocity() * 100
        vect = -np.array([position_target[0] - position[0],
                          position_target[1] - position[1],
                          position_target[2] - position[2]])
        phi_1 = self.mathem.angle_between(velocity, vect)
        phi_1 -= total_lead_angle
        if (t > T_d or (phi_1 > math.radians(270) or phi_1 > math.radians(180))):
            return True
        else: 
            return False

    def start(self):

        group_colors = np.ones((2, 4), dtype=np.float32)
        group_colors[0] = [1, 0, 0, 1]
        group_colors[1] = [0, 0, 1, 1]
        array_position_airplane = np.ones((2, 3), dtype=np.float32)
        N_success = 0
        N_k = 10 # кол-во итераций

        for N in range(0, N_k):
            self.update_init_param()
            array_position_airplane[0] = self.aircraft.get_position()
            array_position_airplane[1] = self.aircraft_target.get_position()

            position_target_airplane = self.aircraft.get_position() + (self.rotate_matrix(self.aircraft.get_phi()) @ self.aircraft_target.get_position()) # координаты связанные с самолётом

            print(self.aircraft_target.get_position())
            print(array_position_airplane)

            time.sleep(1)
            time_scan = 0
            phi = 0
            for i in np.arange(0, 100, self.dt):
                time_scan += self.dt
                current_position_aircraft = self.aircraft.get_position()
                current_position_aircraft_target = self.aircraft_target.get_position()
                # не на кажлом шаге
                # если входи взону без оьшибок, то вероятность поражения  = 1
                # с ошибками. ОШИБКА ПО КУРСУ сигма - 3 процента 6 процентов 9 процентов
                if (time_scan % 10 <= 0.1 or time_scan == self.dt):
                    phi = self.mathem.get_angle(center = np.array([0, 0]), point =  current_position_aircraft_target - current_position_aircraft)
                    phi += math.pi
                    update_errors()
                else:
                    reset_errors()

                # print("phi = ", math.degrees(phi))
                
                vis.get_orientation(phi)
                
                self.aircraft.calculate_position(speed_aircraft, phi + math.radians(90) + total_lead_angle, i)
                self.aircraft_target.calculate_position(speed_aircraft_target, math.radians(90), i)

                position_target_airplane = current_position_aircraft + (self.rotate_matrix(self.aircraft.get_phi()) @ current_position_aircraft_target)
                array_position_airplane[0] = current_position_aircraft
                array_position_airplane[1] = current_position_aircraft_target
                
                # print(array_position_airplane)
                # print("position airplane = ",position_target_airplane)
                vis.setVehicleStates(array_position_airplane, group_colors)

                if (self.is_detection(current_position_aircraft_target, current_position_aircraft, phi + total_lead_angle)):
                    print("time =" , i)
                    N_success += 1
                    break
                else:
                    if (self.is_continue(time_scan, phi)):
                        break

                print("time = ", time_scan)

                # time.sleep(self.dt)

                

        print("W_дн = ", N_success / N_k)


# v1 = np.array([1, 0, 0])
# v2 = np.array([1.1, -0.5, 0])


# mathem = Math()

# print(mathem.angle_between(v1, v2))

# for i in range(0, 10):

simulator = Simulator()

rcv_thread = threading.Thread(target=simulator.start,
                                    name="rcv_thread")

rcv_thread.start()

vis.startVisualization()

