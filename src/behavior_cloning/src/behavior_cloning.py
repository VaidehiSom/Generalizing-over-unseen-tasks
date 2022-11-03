#!/usr/bin/env python

import rospy
from std_msgs.msg import String # TODO: Check data type of subscribers/publishers and import here
from sensor_msgs.msg import JointState
import rosbag

class DataCollection:
    def __init__(self):
        # subscriber, publisher
        # TODO: subscribe- edd effector position, camera feed
        self.end_effector_pos_subsciber = rospy.Subscriber('/your_robot_name/joint_state', JointState, self.end_effector_pos_cb)
        self.camera_feed_subscriber = rospy.Subscriber('', String, self.camera_feed_cb)
        self.file = open("data_collection.txt", "a")
        self.bag = rosbag.Bag("data_collection.bag", 'a')
        print("DataCollection")

    def __del__(self):
        self.file.close()
        self.bag.close()
        print("~DataCollection")

    def end_effector_pos_cb(self, data):
        self.end_effector_pos = data
    
    def camera_feed_cb(self, data):
        self.camera_feed = data

    def record_data(self):
        self.bag.write('/your_robot_name/joint_state', self.end_effector_pos)
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
