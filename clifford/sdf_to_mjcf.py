import xml.etree.ElementTree as ET

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

for child in root[WORLD][MODEL]:    
    curr_i = 0
    if child.tag == "link":
        # We're at the link, save the name
        link_name = child.attrib['name']
        print(link_name)
        
        for grandchild in child:
            if grandchild.tag == "pose":
                # TODO: [Function 1] We need to split this into 3 and 3 (translation and rotation)
                link_pose = grandchild.text
                print(link_pose)

            if grandchild.tag == "inertial":
                # TODO: [Function 1] We need to split this into 3 and 3 (translation and rotation)
                inertial_pose = grandchild[INERTIAL_POSE].text
                inertial_mass = grandchild[INERTIAL_MASS].text
                print(inertial_pose)
                print(inertial_mass)
            
            elif grandchild.tag == "visual":
                geometry = grandchild[VISUAL_GEOMETRY]
                for g_kid in geometry:
                    if g_kid.tag == "mesh":
                        # TODO: [Function 2] We need to strip uri:// prefix from this
                        uri = g_kid[MESH_URI].text
                        print(uri)

                    elif g_kid.tag == "cylinder":
                        radius = g_kid[0].text
                        length = g_kid[1].text

    elif child.tag == "joint":
        print("Done parsing link")
        exit()
    print()
    
