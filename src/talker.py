#!/usr/bin/env python

import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

import numpy
def talker():
    pub = []
    for i in range(4):
        pub_i = rospy.Publisher('ctrl_cmd'+str(i), numpy_msg(Floats),queue_size=10)
        pub.append(pub_i)
    rospy.init_node('talker', anonymous=True)
    r = rospy.Rate(10) # 10hz
    tmp = [1.0, 2.1, 3.2]
    while not rospy.is_shutdown():
        for i in range(4):
            tmp = [i, tmp[0]+0.1,tmp[1]+0.1,tmp[2]+0.1]
            a = numpy.array(tmp, dtype=numpy.float32)
            pub[i].publish(a)
        r.sleep()

if __name__ == '__main__':
    talker()
