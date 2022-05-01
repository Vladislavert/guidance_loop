from turtle import position
from vispy import app, scene, visuals
from vispy.visuals.transforms import STTransform, MatrixTransform

import threading
import time

import numpy as np
import matplotlib.pyplot as plt 
from numpy import linalg as LA
import math



# params
scale_factor = 0.0001
# params radar
detection_range = 80000.0 * scale_factor
total_lead_angle = math.radians(11)
opening_angle = math.radians(20)



# -------------------Graphics----------------------

drawTrajectoryLines = True

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
        # # its acutally an inverted xaxis
        zax.transform = scene.transforms.MatrixTransform()
        zax.transform.rotate(90, (0, 1, 0))  # rotate cw around yaxis
        zax.transform.rotate(-45, (0, 0, 1))  # tick direction towards (-1,-1)

        self.sc3d = Scatter3D(parent=self.view.scene)
        self.sc3d_1 = Scatter3D(parent=self.view.scene)
        self.sc3d.set_gl_state('translucent', blend=True, depth_test=True)


        self.range = scene.visuals.Ellipse(center = [0, 0, 0], color = [0, 1, 1, 0.1], radius = [detection_range, detection_range], border_color = [0, 1, 1, 1])

        self.direction = scene.visuals.Axis(pos=[[0, 0], [0, 5]], tick_direction=(-1, 0),
                         axis_color='g', tick_color='g', text_color='g',
                         font_size=16, parent=self.view.scene, domain=(0., 5.))

        self.view.add(self.range)


        # params aircrafts
        self.orientation = 0

# Box(width=self.bodyBoxWidth,
#                                     height=self.bodyBoxHeight,
#                                     depth=self.bodyBoxDepth,
#                                     color=(0, 0, 1, 1),
#                                     edge_color='green')

        self.group_colors = np.ones((1, 4), dtype=np.float32)
        self.group_poses = np.zeros((1, 3), dtype=np.float32)
        self.update_data = False

    def startVisualization(self):
        # добавляем таймер и запускаем приложение
        timer = app.Timer()
        timer.connect(self._updatePlots)
        timer.start()
        self.canvas3d.show()
        # self.canvasXY.show()
        app.run()
    

    def setVehicleStates(self, pose_list, color_list):
        # так должен выглдеть массив для 4 БЛА
        # poses = np.zeros((4, 3))
        # colors = np.zeros((4, 4))
        # for i in range(len(pose_list)):
        #     pos = n
        self.group_poses = pose_list
        self.group_colors = color_list
        self.update_data = True
    
    def get_orientation(self, orientation):
        self.orientation = orientation

    def _updatePlots(self, ev):
        if(self.update_data == True):
            self.sc3d.set_data(self.group_poses, face_color=self.group_colors, symbol='o', size=10,
                edge_width=0.5, edge_color='blue')
            # self.sphere3D.set_data(self.group_poses, face_color=self.group_colors, symbol='o', size=10,
            #     edge_width=0.5, edge_color='blue')
            # self.sphere3D.set_data(center = self.group_poses[0])
            # self.sc3d_1.set_data(centreCircle)

            tr_range = MatrixTransform()
            tr_range.translate((self.group_poses[0][0],
                          self.group_poses[0][1],
                          self.group_poses[0][2]))
            self.range.transform = tr_range


            tr_direction = MatrixTransform()
            # tr_direction.rotate(math.degrees(self.orientation[0]), (1, 0, 0))
            # tr_direction.rotate(math.degrees(self.orientation[1]), (0, 1, 0))
            tr_direction.rotate(math.degrees(self.orientation), (0, 0, 1))

            tr_direction.translate((self.group_poses[0][0],
                                self.group_poses[0][1],
                                self.group_poses[0][2]))



            self.direction.transform = tr_direction

            # tr.rotate(radToDeg * (bodyOrientation[0]), (1, 0, 0))
            # tr.rotate(radToDeg * (bodyOrientation[1]), (0, 1, 0))
            # tr.rotate(radToDeg * (bodyOrientation[2]), (0, 0, 1))
            
            # self.sphere3D = Sphere3D(parent=self.view.scene, center = self.group_poses[0])
            

            self.update_data = False


vis = Visualizer()

# -------------------------------------------------


class Math:
    def calculate_angle(self, x, y):
        ret_angle = np.arctan2(y , x) + math.radians(180)

        return (ret_angle)


class Math_model_aircraft:
    def __init__(self, velocity, position, phi):

        self.velocity = velocity
        self.position = position
        self.past_velocity = np.array([0, 0, 0])
        self.phi = phi


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
        self.velocity[0] = abs_velocity * np.cos(phi)
        self.velocity[1] = abs_velocity * np.sin(phi)
        self.velocity[2] = 0

    def get_velocity(self):
        return self.velocity

    def get_position(self):
        return self.position
        
    def get_phi(self):
        return self.phi

    def calculate_angle(self, phi):
        self.phi = phi





class Simulator:
    def __init__(self):

        self.mathem = Math()

        



        # params aircraft
        # self.velocity_airplane = np.array([0.0, 300.0, 0])
        self.velocity_airplane = np.array([0, 0, 0]) * scale_factor
        self.position_airplane = np.array([0.0, 0.0, 0]) * scale_factor
        self.phi = 0

        # params aircraft target
        self.position_target = np.array([-75000.0, 250000.0, 0]) * scale_factor
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


    # def calculate_angle(self, x, y):
    #     ret_angle = np.arctan(y / x)

    # def calculate_angle(self, x, y):
    #     ret_angle = np.arctan2(y , x) + math.radians(180)

        return ret_angle

    # def rotate_matrix(self, phi):
    #     array_matrix_2d = np.array([[np.cos(phi), +np.sin(phi)],
    #                             [-np.sin(phi), np.cos(phi)]])


    def rotate_matrix(self, phi):
        array_matrix_2d = np.array([[np.cos(phi),   -np.sin(phi),   0],
                                    [+np.sin(phi),  np.cos(phi),    0],
                                    [0,             0,              1]])
        
        return array_matrix_2d

    def is_detection(self):
        distance =  np.array([0.0, detection_range])
        if (LA.norm(self.position_target) <= LA.norm(self.position_airplane + self.rotate_matrix(self.phi + self.total_lead_angle) @ distance)):
            angle_vector = self.angle(self.rotate_matrix(self.phi) @ self.position_airplane, self.position_target)
            print("radius")
            if (angle_vector >= 1.57 - 0.34 and angle_vector >= 1.57 - 0.34):
                print("angle =", angle_vector * 180 / 3.14)
                return (True)
            else:
                return (False)
        else:
            return (False)

    def start(self):
        
        # array_position_airplane = []
        array_position_target = []

        group_colors = np.ones((2, 4), dtype=np.float32)
        array_position_airplane = np.ones((2, 3), dtype=np.float32)

        array_position_airplane[0] = self.aircraft.get_position()
        array_position_airplane[1] = self.aircraft_target.get_position()


        position_target_airplane = [] # координаты связанные с самолётом

        print(self.aircraft_target.get_position())
        print(array_position_airplane)

        time.sleep(1)
        for i in np.arange(0, 100, self.dt):
            
            # if (is_detection(position_target, position_airplane, phi)):
            #     print("time =" , i)
            #     break


            phi = self.mathem.calculate_angle(array_position_airplane[1][1] , array_position_airplane[1][0])
            print("phi = ", math.degrees(phi))

            vis.get_orientation(math.radians(180) - phi)
            
            # phi -= 0.001

            # phi = calculate_angle(position_target[0], position_target[1])

            

            self.aircraft.calculate_position(300 * scale_factor, phi, i)
            self.aircraft_target.calculate_position(-300 * scale_factor, self.aircraft_target.get_phi(), i)

            current_position_aircraft = self.aircraft.get_position()
            current_position_aircraft_target = self.aircraft_target.get_position()

            position_target_airplane = current_position_aircraft + (self.rotate_matrix(self.aircraft.get_phi()) @ current_position_aircraft_target)

            array_position_airplane[0] = current_position_aircraft
            array_position_airplane[1] = current_position_aircraft_target
            # print(current_position_aircraft)
            print(array_position_airplane)
            print("position airplane = ",position_target_airplane)
            vis.setVehicleStates(array_position_airplane, group_colors)
            time.sleep(self.dt)



simulator = Simulator()

rcv_thread = threading.Thread(target=simulator.start,
                                      name="rcv_thread")
rcv_thread.start()

vis.startVisualization()



# show(0)

# plt.plot(position_airplane[0], position_airplane[1], 'o')


# plt.plot(array_position_airplane[0], array_position_airplane[1])
# for i in range(0, len(array_position_target)):
# plt.plot(array_position_target, 'b')
# plt.show()

