"""IO tools for mp0.
"""


def read_data_from_file(filename):
    """
    Read txt data from file.
    Each row is in the format article_id\ttitle\tpositivity_score\n.
    Store this information in a python dictionary. Key: article_id(int),
    value: [title(str), score(float)].

    Args:
        filename(string): Location of the file to load.
    Returns:
        out_dict(dict): data loaded from file.
    """
    out_dict = {}
    file = open(filename, "r")

    for line in file:
        parts = line.split('\t')
        parts[0] = int(parts[0]) #Converting first word(article_id) to integer
        parts[2] = parts[2][:-1] #Removing the last character '\n'
        parts[2] = float(parts[2]) #Converting score to float
        values = [parts[1], parts[2]] 
        out_dict[parts[0]] = values
    
    return out_dict


def write_data_to_file(filename, data):
    """
    Writes data to file in the format article_id\ttitle\tpositivity_score\n.

    Args:
        filename(string): Location of the file to save.
        data(dict): data for writting to file.
    """
    file = open(filename, "w")
    for key in data:
    	file.write(str(key)+'\t'+str(data[key][0])+'\t'+str(data[key][1])+'\n')

    file.close()
    pass
