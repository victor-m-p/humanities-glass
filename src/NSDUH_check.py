import pandas as pd 
import numpy as np 

# try just pandas

# load (huge) file 
infile = '../data/raw/NSDUH_2020_Tab.txt'
d = pd.read_csv(infile, sep='\t')
d['ID'] = d.index

# categories
c_miscsubstance = ['ID', 'CIGEVER', 'ALCEVER', 'MJEVER', 
                   'COCEVER', 'CRKEVER', 'HEREVER'] 
c_hallucinogenics = ['ID', 'LSD', 'PCP', 'PEYOTE', 'MESC', 
                     'PSILCY', 'ECSTMOLLY', 'KETMINESK',
                     'DMTAMTFXY', 'SALVIADIV', 'HALLUCOTH']
c_inhalants = ['ID', 'AMYLNIT', 'CLEFLU', 'GAS', 'ETHER',
               'SOLVENT', 'LGAS', 'NITOXID', 'FELTMARKR',
               'SPPAINT', 'AIRDUSTER', 'OTHAEROS', 'INHALOTH']
c_depression = ['ID', 'ADDPREV', 'ADDSCEV', 'ADLOSEV', 'ADDPDISC',
                'ADDPLSIN', 'ADDSLSIN', 'ADLSI2WK', 
                'ADDPR2WK', 'ADDPPROB']

# subsets
d_miscsubstance = d[c_miscsubstance]
d_hallucinogenics = d[c_hallucinogenics]
d_inhalants = d[c_inhalants]
d_depression = d[c_depression]

d_hallucinogenics.groupby('LSD').size()

# recode dict 
dct_recode = {
    85: 0, # bad data
    91: 2, # NEVER (same as NO here)
    94: 0, # DK
    97: 0, # refused 
    98: 0 # blank/no answer
}

# miscsubstances 
for col in d_miscsubstance.columns: 
    if not col == 'ID': 
        d_miscsubstance.replace(
            {col: dct_recode}, 
            inplace = True)

# hallucinogenics
# this is a bit slow
import itertools 
def all_equal(row): 
    g = itertools.groupby(row)
    return next(g, 1) and not next(g, 2)

vals_hallucinogenics = []
ids_hallucinogenics = []
for index in range(len(d_hallucinogenics)):
    rowlst = d_hallucinogenics.loc[index, c_hallucinogenics[1:]].values.flatten().tolist()
    allEquals = all(element == 91 for element in rowlst)
    vals_hallucinogenics.append(allEquals)
    ids_hallucinogenics.append(index)

# pandas 
d_hallucinogenics = pd.DataFrame({
    'ID': ids_hallucinogenics,
    'hallucinogenics': vals_hallucinogenics
})

# recode 
d_hallucinogenics.replace(
    {'hallucinogenics': {False: 1, True: 2}}, # i.e. 1 = yes, 2 = no 
    inplace = True)

# inhalants (same pipeline)


# CODING:  
# https://www.datafiles.samhsa.gov/sites/default/files/field-uploads-protected/studies/NSDUH-2020/NSDUH-2020-datasets/NSDUH-2020-DS0001/NSDUH-2020-DS0001-info/NSDUH-2020-DS0001-info-codebook.pdf

# SECTION: SELF-ADMINISTERED SUBSTANCE USE #
#### general substance #####
## cigarettes: CIGEVER (1 = yes, 2 = no)
## alcohol: ALCEVER (1 = yes, 2 = no, others = DK/NA)
## marijuana: MJEVER (1 = yes, 2 = no, others = DK/NA)
## cocaine: COCEVER (1 = yes, 2 = no, others = DK/NA)
## crack: CRKEVER (1 = yes, 2 = no, 91, others = DK/NA)
## heroin: HEREVER (1 = yes, 2 = no, others = DK/NA)


#### HALLUCINOGENICS (last ones coded a bit differently I think) ####
### lsd: LSD (91 overal hallucinogenics)
### pcp: PCP: (same as above)
### peyote: PEYOTE (same as above)
### mescaline: MESC (same as above)
### psilocybin: PSILCY (same as above)
### Ecstacy: ECSTMOLLY (same as above)
## Ketamine: KETMINESK (91 specific Ketamine?)
## DMT/AMT/Foxy: DMTAMTFXY (91 specific DMT/AMT/FOXY)
## Salvia divinorum: SALVIADIV (91 specific SALVIADIV)
## ever other hallucinogens? HALLUCOTH (91 overall hallucinogenics)

#### INHALANTS ####
## Amyl Nitrite: AMYLNIT (91 overall inhalants)
## correction fluid, etc: CLEFLU (-//-)
## gas: GAS (-//-)
## glue: GLUE (-//-)
## ...: ETHER (-//-)
## ...: SOLVENT (-//-)
## ...: LGAS (-//-)
## ...: NITOXID (-//-)
## ...: FELTMARKR (-//-)
## ...: SPPAINT (-//-)
## ...: AIRDUSTER (-//-)
## ...: OTHAEROS (-//-)
## ...: INHALOTH (-//-)

#### METH ####
## methamphetamine: Methamever (1, 2, DK/NA)

#### Pain relievers #### 
## oxycontin past 12mo: OXCNANYYR (1, 2, 81, 91, 93, 95)
### 81 = never used pain relievers (logically assigned)
### 91 = never used pain relievers 
### 93 = did not use in past 12mo
### 95 = not any pain relievers in past 12mo
## pain reliever lifetime: PNRANYLIF (1, 2, 5, ...)

#### Tranquilizers #####
## TRQANYLIF: (1, 2, 5, ...)

#### Stimulants ####
## STMANYLIF (1, 2, 5, ...)

#### Sedatives ####
## SEDANYLIF (1, 2, 5, ...)

######## a lot of stuff left out here starting p. 64 ########
# SECTION: OTHER SELF-ADMINISTERED SECTIONS #
## SUBSECTION: Adult depression 
### ADDPREV (1, 2, 97, 99)
### ADDSCEV (-//-)
### ADLOSEV (-//-)
### ADDPDISC (-//-)
### ADDPLSIN (-//-)
### ADDSLSIN (-//-)
### ADLSI2WK (-//-)
### ADDPR2WK (-//-)
### ADDPPROB (-//-)
###### Just keeps on going.... (stopped)
###### NB: 99 (legitimate skip) varies a lot
###### I thought it was people under 18, but??
###### Also some non-binary variables here. 

##### in general ##### 
## 1 = yes
## 2 = no
## 3 = yes (logically assigned) -- so recode to 1
## 5 = yes (specific) -- so recode to 1
## 85 = bad data
## 91 = NEVER USED (so recode to 2 in some cases)
## 94 = DK
## 97 = refused 
## 98 = blank / no answer