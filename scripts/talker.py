#!/usr/bin/env python3
# license removed for brevity
# import rospy
# from std_msgs.msg import String

# def talker():
#     pub = rospy.Publisher('chatter', String, queue_size=10)
#     rospy.init_node('talker', anonymous=True)
#     rate = rospy.Rate(10) # 10hz
#     while not rospy.is_shutdown():
#         hello_str = "hello world %s" % rospy.get_time()
#         rospy.loginfo(hello_str)
#         pub.publish(hello_str)
#         rate.sleep()

# if __name__ == '__main__':
#     try:
#         talker()
#     except rospy.ROSInterruptException:
#         pass

import mysql.connector

mydb = mysql.connector.connect(
    host = "10.1.15.238",
    user = "admin",
    password = "admin",
    database = "test"
    )

mycursor = mydb.cursor()
sql = "INSERT INTO employees (name, age, country, position, wage) VALUES (%s, %s, %s, %s, %s)"
val = ("late", 21, "thailand", "changmai", 40000)

mycursor.execute(sql, val)
mydb.commit()

print(mycursor.rowcount, "record inserted")