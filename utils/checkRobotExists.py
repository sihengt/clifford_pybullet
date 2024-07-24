def checkRobotExists(func):
    def decoratedFunc(*args, **kwargs):
        if args[0].robotID == None:
            print("ERROR: no robotID.")
            return
        elif args[0].nJoints == None:
            print("ERROR: nJoints not populated from sdf.")
            return
        return func(*args, **kwargs)
    return decoratedFunc