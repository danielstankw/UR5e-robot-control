import xml.etree.ElementTree as ET
import lxml

tree =  ET.parse('/home/danieln7/Desktop/RobotCodeDaniel/rtdeConfigurationDefault.xml')
root = tree.getroot()


for send in root.findall("./send[@key='in']"):
    for field in send.iter('field'):
        print(field.attrib['name'])