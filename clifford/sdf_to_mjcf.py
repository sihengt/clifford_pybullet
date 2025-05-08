import xml.etree.ElementTree as ET
from xml.dom import minidom

WORLD = 0
MODEL = 0

tree = ET.parse("clifford.sdf")
root = tree.getroot()

# names = []
# pos = []
data = {}

INERTIAL_POSE = 0
INERTIAL_MASS = 1
VISUAL_GEOMETRY = 1
MESH_URI = 1

## QUESTIONS
# 1: How to port cylinder over to MuJoCo for the tires?
# 2: How to store the information we need? Or should we construct the MJCF 
    
# All the links:
a = ET.Element('mujoco')
b = ET.SubElement(a, 'compiler')
c = ET.SubElement(a, "asset")
wb = ET.SubElement(a, "worldbody")
e = ET.SubElement(wb, "geom", {"name":"chassis", "type":"mesh", "mesh":"body"})
b = ET.SubElement(wb, "body", {"name":"body", "pos":"0 0 .03"})

def split_pose(pose):
    pose_split = pose.split()
    translation = " ".join(pose_split[:3])
    orientation = " ".join(pose_split[3:])
    return translation, orientation

for neighbor in root.iter("link"):
    link_name = neighbor.attrib['name']

    link_pose = neighbor.find("pose").text
    link_t, link_o = split_pose(link_pose)
    curr_link = ET.SubElement(b, "body", {"name":link_name, "pos":link_t, "euler":link_o})

    inertial_pose = neighbor.find("inertial").find("pose").text
    inertial_t, inertial_o = split_pose(inertial_pose)
    inertial_mass = neighbor.find("inertial").find("mass").text

    geometry = neighbor.find("visual").find("geometry")
    mesh = geometry.find("mesh")
    if mesh:
        uri = mesh.find("uri").text
        uri_split = uri.split("//")
        mesh = uri_split[-1]
        ET.SubElement(curr_link, "geom", {"name":link_name, "type":"mesh", "mesh":mesh})

    elif geometry.find("cylinder"):
        radius = geometry.find("cylinder").find("radius").text
        length = geometry.find("cylinder").find("length").text
        ET.SubElement(curr_link, "geom", {"name":link_name, "type":"cylinder", "size":" ".join([radius, length])})
    
    else:
        print("ERROR")

    ET.SubElement(curr_link, "inertial", {"pos":inertial_t, "euler":inertial_o, "mass":inertial_mass})


def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, encoding="utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

with open("output.xml", "w", encoding="utf-8") as f:
    f.write(prettify(a))

