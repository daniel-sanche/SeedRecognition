import pickle

"""
creates a label dict file that holds the names of each folder associated with each class
"""
def generate_indices(datasetDir, config_name="slice_config", indexFileName="labelDict.p"):
    config_path = datasetDir + "/" + config_name
    labelDict = {}
    thisLabel = -1
    with open(config_path) as f:
        for line in f:
            number, foldername = line.rstrip().split()
            if number == '1':
                thisLabel += 1
            oldList = labelDict.get(thisLabel, [])
            labelDict[thisLabel] = oldList + [foldername]
    with open(indexFileName, 'wb') as file:
        pickle.dump(labelDict, file, protocol=pickle.HIGHEST_PROTOCOL)
    return labelDict

"""
generator that returns the next folder name for each class
"""
def getFolderSet(labelDict):
    batchDict = []
    while True:
        for i in range(10):
            for key in labelDict:
                batchDict.append(labelDict[key][i])
            yield batchDict
            batchDict = []


if __name__ == "__main__":
    dataset_path = "/Users/Sanche/Datasets/Seeds_Full"
    labelDict = generate_indices(dataset_path)

    for i in getFolderSet(labelDict):
        print (len(i))