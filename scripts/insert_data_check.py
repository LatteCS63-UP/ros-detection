#!/usr/bin/env python3
# license removed for brevity

import rospy
from std_msgs.msg import String

from datetime import datetime
import mysql.connector

def insert(data_insert):
    # date and time
    dateAndTime = datetime.now()

    # Date
    year = dateAndTime.strftime("%Y")
    month = dateAndTime.strftime("%m")
    day = dateAndTime.strftime("%d")

    date = f"{year}-{month}-{day}"

    # Time
    hour = dateAndTime.strftime("%H")
    minute = dateAndTime.strftime("%M")
    second = dateAndTime.strftime("%S")

    time = f"{hour}:{minute}:{second}"

    rospy.loginfo(str(data_insert.data))

    mycursor = mydb.cursor()
    sql = "INSERT INTO checkname (StudentID, SubjectID, DateCheck, TimeCheck, status) VALUES (%s, %s, %s, %s, %s)"
    val = (data_insert.data, "225382", date, time, "มาเรียน")

    mycursor.execute(sql, val)
    mydb.commit()

    rospy.loginfo(str(mycursor.rowcount) + " record inserted")

def insert_check():
    rospy.init_node("insert_dataToMysql", anonymous=True)

    rospy.Subscriber("name_detection", String, insert)

    rospy.spin()

if __name__ == "__main__":

    mydb = mysql.connector.connect(
        host = "10.1.15.62",
        user = "admin",
        password = "admin",
        database = "project_robot_checking"
    )

    insert_check()