from deeplearning_ai.kaggle_problems.titanic import init as titanic_init

if __name__ == '__main__':
    if len(sys.argv) > 1:
        command = str(sys.argv[1])
        if command == "titanic":
            titanic_init()
        else
            print("command not found")
    else:
        print("command arg is empty")
