import argparse
import pandas as pd

mirrorsURL = pd.read_csv("scripts/mirrorsURL.csv")

def replaceURL(oldURL,newURL,filename="CMakeLists.txt"):
    # Safely read the input filename using 'with'
    with open(filename) as f:
        s = f.read()
        urlNum=len(mirrorsURL[oldURL])
        oldurl_not_in_s=0
        for old_string in mirrorsURL[oldURL]:
            if old_string not in s:
                oldurl_not_in_s+=1
        if oldurl_not_in_s==urlNum:
            print("No need to replace")
            return False
    # Safely write the changed content, if found in the file
    with open(filename, 'w') as f:        
        for rowIdx in mirrorsURL[oldURL].index:
            old_string=mirrorsURL.loc[rowIdx,:][oldURL]
            new_string=mirrorsURL.loc[rowIdx,:][newURL]    
            print('Changing "{old_string}" to "{new_string}" in {filename}'.format(**locals()))        
            s = s.replace(old_string, new_string)
        f.write(s)
    return True

def main_function():
    parser = argparse.ArgumentParser(description='hook to change FetchContent_Populate in CMakefile.txt',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=
    '''
    Pls cite the 
    work.

    ''')
    parser.add_argument('-p','--pubURL',action='store_true', help='Use public repos')
    parser.add_argument('-k','--liukURL', action='store_true', help="Use liuk's repos")
    args = parser.parse_args()
    dArgs = vars(args)
    for aname,v in dArgs.items():
        if v:
            newurl=aname
        else:
            oldurl=aname
    if newurl!=oldurl:
        replaceURL(oldurl,newurl)

if __name__ == '__main__':
    main_function()
    
