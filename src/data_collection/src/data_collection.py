#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image # TODO: Check data type of subscribers/publishers and import here
from sensor_msgs.msg import JointState
import rosbag

class DataCollection:
    def __init__(self):
        # subscriber, publisher
        # TODO: subscribe- edd effector position, camera feed
        self.joint_states_sub = rospy.Subscriber('/my_gen3/joint_states', JointState, self.joint_states_cb)
        self.camera_feed_color_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.camera_color_feed_cb)
        self.camera_feed_depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.camera_depth_feed_cb)
        self.file = open("data_collection.txt", "a")
        self.bag = rosbag.Bag("data_collection.bag", 'a')
        print("DataCollection")

    def __del__(self):
        self.file.close()
        self.bag.close()
        print("~DataCollection")

    def joint_states_cb(self, data):
        """header: 
            seq: 1774
            stamp: 
                secs: 1668551773
                nsecs: 509788036
            frame_id: ''
          name: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7, finger_joint, left_inner_knuckle_joint,
          left_inner_finger_joint, right_outer_knuckle_joint, right_inner_knuckle_joint, right_inner_finger_joint]
          position: [-0.005732187930286514, 0.4887219180940803, -3.132436439445992, -1.864200246198343, 0.0024580001582711858, 0.7828533132387959, 1.5643789075161325, 0.00698692093770088, 0.00698692093770088, -0.00698692093770088, 0.00698692093770088, 0.00698692093770088, -0.00698692093770088]
          velocity: [-0.002289906319662698, 0.17106830242405116, 0.010021302492616237, 0.3409748008225553, 0.0042357197429195486, -0.1534153726908928, -0.00755036961837568, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0]
          effort: [-0.31865596771240234, 17.228782653808594, 0.03316592797636986, -10.732674598693848, -0.0162641778588295, -2.255537509918213, -0.31152278184890747, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        """
        self.end_effector_pos = data
    
    def camera_depth_feed_cb(self, data):
        self.camera_depth_feed = data

    def camera_color_feed_cb(self, data):
        self.camera_color_feed = data

    def record_data(self):
        self.bag.write('/my_gen3/joint_states', self.end_effector_pos)
        self.bag.write('', self.camera_feed)



if __name__ == '__main__':
    rospy.init_node('test_node', disable_signals=True)
    data_c = DataCollection()
    # rate = rospy.Rate(10)
    timer_for_one_round_mins = 1 # 1 min
    a = input("How many rounds?")

    for i in range(a):
        now = rospy.get_rostime()
        while(rospy.Time.now() != now + rospy.Duration(timer_for_one_round_mins*60)):
            data_c.record_data()
            rospy.sleep(0.01)
        
        ans = input("Save this data?")
        if(ans=="y" or ans=="Y"):
            with rosbag.Bag('final_data.bag', 'w') as outbag:
                for topic, msg, t in rosbag.Bag('data_collection.bag').read_messages():
                    outbag.write(topic, msg, t)
        rospy.sleep(0.1)

    rospy.spin()

    # rospy.signal_shutdown # TODO: Check how to shutdown this

    # while not rospy.is_shutdown():
    #     rospy.sleep(0.01)
