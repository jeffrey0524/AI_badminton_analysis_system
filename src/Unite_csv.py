'''
    this file is use for unite csv label
    becouse master find out that some label in csv is 'BallLocationX' instad of 'LandingX'
    and this file can rename 'BallLocationX' to 'LandingX'
'''
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parents[1]
import pandas as pd
import glob

def csv_Unite(dir_path: str):
    filenames = get_filenames(dir_path, '*S2.csv')
    filenames.sort()
    for filename in filenames:
        df = pd.read_csv(filename)

        df.rename(columns={'BallLocationX': 'LandingX'}, inplace=True)
        df.rename(columns={'BallLocationY': 'LandingY'}, inplace=True)
        df.columns = df.columns.str.replace(' ', '')
        df.to_csv(filename, index=False)
    pass

def get_filenames(dir_path: str, specific_name: str, withDirPath=True) -> list:
    '''
    get_filenames
    -----
    This function can find any specific name under the dir_path, even the file inside directories.

    specific_name:
        >>> Can type any word or extension.
        e.g. '*cat*', '*.csv', '*cat*.csv',
    '''

    if dir_path[-1] != '/':
        dir_path += '/'

    filenames = glob.glob(f'{dir_path}**/{specific_name}', recursive=True)

    if '*.' == specific_name[:2]:
        filenames.extend(glob.glob(f'{dir_path}**/{specific_name[1:]}', recursive=True))

    if withDirPath is False:
        dir_path_len = len(dir_path)
        filenames = [filename[dir_path_len:] for filename in filenames]

    return filenames

if __name__ == '__main__':
    # dir_path = f'{PROJECT_DIR}/part1/train'
    dir_path = f'{PROJECT_DIR}/test_part1/train'
    csv_Unite(dir_path)
    pass