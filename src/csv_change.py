import pandas as pd

# pd.Dataframe
def read_csv(path:str) -> list:
    df = pd.read_csv(path)
    labal = pd.read_csv(path).values.tolist()
    for i in range(len(labal)):
            labal[i][2] = labal[i][2][0]     # remove ' '
    return df.values.tolist()

def write_csv(list:list,path:str)->str:
    df = pd.DataFrame(list)
    df.columns = ['ShotSeq','HitFrame','Hitter','RoundHead','Backhand','BallHeight','LandingX','LandingY','HitterLocationX','HitterLocationY','DefenderLocationX','DefenderLocationY','BallType','Winner']
    df.to_csv(path)
    return df

def write_pickle(list:list,path:str)->str:
    df = pd.DataFrame(list)
    df.columns = ['ShotSeq','HitFrame','Hitter','RoundHead','Backhand','BallHeight','LandingX','LandingY','HitterLocationX','HitterLocationY','DefenderLocationX','DefenderLocationY','BallType','Winner']
    df.to_pickle(path)
    return path

if __name__ == '__main__':
    paths = ['part1/train/00001/00001_S2.csv']
    for path in paths:
        print(read_csv(path))
        # write_csv(read_csv(path),'src/123.csv')
        # write_pickle(read_csv(path),'src/123.pickle')
        continue
