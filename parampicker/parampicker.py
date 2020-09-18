import argparse
import datetime
import json
import logging
import os
import numpy as np
from logging import handlers

logger = logging.getLogger('experiment')


# How to use this class (See the main function at the end of this file for an actual example)

## 1. Create an parampicker object in the main file (the one used to run the parampicker)
## 2. Pass a name and args_parse objecte. Output_dir corresponds to directly where all the results will be stored
## 3. Use parampicker.path to get path to the output_dir to store any other results (such as saving the model)
## 4. You can also store results in parampicker.result dictionary (Only add objects which are json serializable)
## 5. Call parampicker.store_json() to store/update the json file (I just call it periodically in the training loop)

class ParamPicker:
    '''
    Class to create directory and other meta information to store parampicker results.
    A directory is created in output_dir/DDMMYYYY/name_0
    In-case there already exists a folder called name, name_1 would be created.

    Race condition:
    '''

    def __init__(self, name, args, rank):
        import sys

        self.all_params = args
        total_seeds = len(args["seed"])
        args = self.get_run(args, rank)
        if "output_dir" in args:
            output_dir = args['output_dir']
        else:
            output_dir = "../../"

        rank = int(rank/total_seeds)

        self.command_args = "python " + " ".join(sys.argv)

        if not args is None:
            if rank is not None:
                self.name = name+ "/" + str(rank) + "/" + "run"
            else:
                self.name = name
            self.params = args
            print(self.params)
            self.results = {}
            self.dir = output_dir

            root_folder = datetime.datetime.now().strftime("%d%B%Y")

            if not os.path.exists(output_dir + root_folder):
                try:
                    os.makedirs(output_dir + root_folder)
                except:
                    assert (os.path.exists(output_dir + root_folder))

            self.root_folder = output_dir + root_folder
            full_path = self.root_folder + "/" + self.name

            ver = 0

            while True:
                ver += 1
                if not os.path.exists(full_path + "_" + str(ver)):
                    try:
                        os.makedirs(full_path + "_" + str(ver))
                        break
                    except:
                        pass
            self.path = full_path + "_" + str(ver) + "/"

            fh = logging.FileHandler(self.path + "log.txt")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(
                logging.Formatter('rank:' + str(args['rank']) + ' ' + name + ' %(levelname)-8s %(message)s'))
            logger.addHandler(fh)

            ch = logging.handlers.logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(
                logging.Formatter('rank:' + str(args['rank']) + ' ' + name + ' %(levelname)-8s %(message)s'))
            logger.addHandler(ch)
            logger.setLevel(logging.DEBUG)
            logger.propagate = False

            self.store_json()

    def is_jsonable(self, x):
        try:
            json.dumps(x)
            return True
        except:
            return False

    def add_result(self, key, value):
        assert (self.is_jsonable(key))
        assert (self.is_jsonable(value))
        self.results[key] = value

    def store_json(self):
        with open(self.path + "metadata.json", 'w') as outfile:
            json.dump(self.__dict__, outfile, indent=4, separators=(',', ': '), sort_keys=True)
            outfile.write("")

    def get_json(self):
        return json.dumps(self.__dict__, indent=4, sort_keys=True)

    def get_run(self, arg_dict, rank):
        # print(arg_dict)
        combinations = []

        if isinstance(arg_dict["seed"], list):
            combinations.append(len(arg_dict["seed"]))

        for key in arg_dict.keys():
            if isinstance(arg_dict[key], list) and not key == "seed":
                combinations.append(len(arg_dict[key]))

        total_combinations = np.prod(combinations)
        selected_combinations = []
        for base in combinations:
            selected_combinations.append(rank % base)
            rank = int(rank / base)

        counter = 0
        result_dict = {}

        result_dict["seed"] = arg_dict["seed"]
        if isinstance(arg_dict["seed"], list):
            result_dict["seed"] = arg_dict["seed"][selected_combinations[0]]
            counter += 1

        for key in arg_dict.keys():
            if key != "seed":
                result_dict[key] = arg_dict[key]
                if isinstance(arg_dict[key], list):
                    result_dict[key] = arg_dict[key][selected_combinations[counter]]
                    counter += 1

        logger.info("Parameters %s", str(result_dict))

        return result_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='iCarl2.0')
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs2', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lrs', type=float, nargs='+', default=[0.00001],
                        help='learning rate (default: 2.0)')
    parser.add_argument('--decays', type=float, nargs='+', default=[0.99, 0.97, 0.95],
                        help='learning rate (default: 2.0)')
    parser.add_argument('--seed', type=int, nargs='+', default=[0, 1, 2],
                        help='learning rate (default: 2.0)')
    parser.add_argument('--rank', type=int, default=6,
                        help='learning rate (default: 2.0)')

    args = vars(parser.parse_args())
    e = ParamPicker("TestExperiment3", args, args['rank'])
    e.add_result("Test Key", "Test Result")
    e.store_json()

