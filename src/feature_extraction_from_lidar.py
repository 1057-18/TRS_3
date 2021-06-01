#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point

def get_intersect(l1, l2):
    # l1, l2 in standard form ax + by + c = 0
    x, y, z = np.cross(l1, l2)
    return (x/z, y/z)

def calculate_euclidian_coordinates(rho, theta):
    x = [rho[i]*np.math.cos(theta[i]) for i in range(len(rho))]
    y = [rho[i]*np.math.sin(theta[i]) for i in range(len(rho))]
    return np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1)

def distance_between_two_points(x1, y1, x2, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2)**(1/2)

def distance_from_point_to_line(x, y, r):
    return np.abs((r[0]*x + r[1]*y + r[2])/np.sqrt(r[0]**2 + r[1]**2))

def least_squares(X, y):
    X = np.hstack((np.ones((len(X), 1)), X))

    return np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, y))

def split_in_respect_to(ind, x, y):
    x1 = x[0:ind]
    y1 = y[0:ind]
    x2 = x[ind+1:]
    y2 = y[ind+1:]
    dist1 = distance_between_two_points(x1[-1], y1[-1], x[ind-1], y[ind-1])
    dist2 = distance_between_two_points(x2[-1], y2[-1], x[ind-1], y[ind-1])
    if dist1 <= dist2:
        x1 = np.vstack((x1, x[ind-1]))
        y1 = np.vstack((y1, y[ind-1]))
    else:
        x2 = np.insert(x2, 0, x[ind-1], axis=0)
        y2 = np.insert(y2, 0, y[ind-1], axis=0)
    return x1, y1, x2, y2

def split_and_merge(x, y, r_list, theta_list, x_points_list, y_points_list, threshold=0.1):
    eq = least_squares(x[[0, -1]], y[[0, -1]])
    intercept = eq[0][0]
    slope = eq[1][0]
    distances = distance_from_point_to_line(x, y, (-slope, 1, -intercept))
    if np.max(distances) > threshold:
        ind_max = np.argmax(distances)
        x1, y1, x2, y2 = split_in_respect_to(ind_max, x, y)
        split_and_merge(x1, y1, r_list, theta_list, x_points_list, y_points_list)
        split_and_merge(x2, y2, r_list, theta_list, x_points_list, y_points_list)
    else:
        print(f'slope: {slope}')
        print(f'intercept: {intercept}')
        l1 = np.array([-slope, 1, -intercept])
        l2 = np.array([1/slope, 1, 0])
        x_cross, y_cross = get_intersect(l1, l2)
        if x_cross >= 0:
            theta = np.math.atan(-1/slope)
        else:
            theta = np.pi + np.math.atan(-1/slope)
        theta_list.append(theta)
        r = distance_from_point_to_line(0, 0, (-slope, 1, -intercept))
        r_list.append(r)
        x_points_list.append(x)
        y_points_list.append(y)
    return r_list, theta_list, x_points_list, y_points_list

def callback(scan):
    angles = np.arange(0, 2*np.pi, (2*np.pi)/360)
    ranges = np.array(scan.ranges)
    log_vec = ranges != float('inf')
    x, y = calculate_euclidian_coordinates(ranges[log_vec], angles[log_vec])
    r_list, theta_list, x_points_list, y_points_list = split_and_merge(x, y, r_list=[], theta_list=[], x_points_list=[], y_points_list=[])

    pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=100)
    marker_array = MarkerArray()
    index = 0
    for i in range(len(x_points_list)):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.get_rostime()
        marker.ns = "walls"
        marker.id = index = index + 1
        marker.lifetime = rospy.Duration(1)
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        marker.color.r = 1.0
        marker.color.a = 1.0

        for j in range(len(x_points_list[i])):
            point = Point()
            point.x = x_points_list[i][j]
            point.y = y_points_list[i][j]
            point.z = 0.1
            marker.points.append(point)

        marker_array.markers.append(marker)
    pub.publish(marker_array)

    # r_list = np.array(r_list).reshape(-1, 1)
    # theta_list = np.array(theta_list).reshape(-1, 1)
    # print(np.hstack((r_list, theta_list)))
    print('___________________________________________')


if __name__ == '__main__':
    try:
        rospy.init_node('feature_extraction')
        sub = rospy.Subscriber('scan', LaserScan, callback)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass