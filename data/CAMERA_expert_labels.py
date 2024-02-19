import io
import pandas as pd
import msoffcrypto


# Clinical UPDRS labels given by Suzie 
UPDRS_labels_suzie_path = '/mnt/teamshare-camera/CAMERA Booth Data/' + 'CAMERA Study Booth - Tracking Log_RM-No Names.xlsx'
# Load the encrypted Excel file
passwd = 'pprc'
decrypted_workbook = io.BytesIO()
with open(UPDRS_labels_suzie_path, 'rb') as file:
    office_file = msoffcrypto.OfficeFile(file)
    office_file.load_key(password=passwd)
    office_file.decrypt(decrypted_workbook)
UPDRS_labels_suzie = pd.read_excel(decrypted_workbook, sheet_name='SA', header=2, usecols='A,B,Q,S,Z')
UPDRS_labels_suzie = UPDRS_labels_suzie.dropna()
# iterate over and populate the dict
UPDRS_med_data_SA = {}
for i, row in UPDRS_labels_suzie.iterrows():
    date = str(row['Date'])[:10].replace('-', '')
    label = {'hand_movement': {'right_open_close': {date: row['Right_Open_Close_Collection_UPDRS']},
                                 'left_open_close': {date: row['Left_Open_Close_Collection_UPDRS']},},}
    UPDRS_med_data_SA[str(row['ID'])] = label

# Clinical UPDRS labels given by Kye Won Park
UPDRS_labels_kw = pd.read_excel(decrypted_workbook, sheet_name='KW', header=2, usecols='A,B,M,O,')
UPDRS_labels_kw = UPDRS_labels_kw.dropna()
# iterate over and populate the dict
UPDRS_med_data_KW = {}
for i, row in UPDRS_labels_kw.iterrows():
    date = str(row['Date'])[:10].replace('-', '')
    label = {'hand_movement': {'right_open_close': {date: row['Right_Open_Close_Collection_UPDRS']},
                                 'left_open_close': {date: row['Left_Open_Close_Collection_UPDRS']},},}
    UPDRS_med_data_KW[str(row['ID'])] = label

# Samples with under 10 cycles
too_short = {
    '17202': ['left',],
    '21401' : ['left',],
    '23160' : ['right',],
    '24318' : ['left',],
    '28641' : ['left','right'],
    '31240' : ['right',],
    '31848' : ['left',],
    '32282' : ['left',],
    '38215' : ['left',],
    '38256' : ['left','right'],
    '38519' : ['left','right'],
    # '' : ['',],
    # '' : ['',],
    # '' : ['',],

    # '' : [,],
}

very_long = {   # end 1, end 2, ...
    '28411' : ['left',],        # 250, 505
    '28615' : ['right',],       # 110, 250, 350, 475, 600, 715
    '28731' : ['left',],        # 400, 900
    '28813' : ['right',],       # 500, 1100
    '30104' : ['left',],        # 390, 705
    '32519' : ['left','right'], # (260, 525), (225, 450,)
    '33164' : ['left',],        # 180, 280, 380, 480, 600
    '34965' : ['right',],       # 205, 375, 575, 790, 1025,  
    '35623' : ['left','right'], # (180, 360, 550,), (150, 325, 510,)
    '35747' : ['left',],        # 175, 360, 575, 775, 1000
    '36220' : ['left',],        # 300, 675
    '36297' : ['left','right'], # (300, 525,), (300, -1)
    '36436' : ['left',],        # 150, 300, 425, 575, 750, -1
    '38215' : ['left',],        # 260, 525
    '38256' : ['left','right'], # (105, 215, 350), (300, 450, 625, 800, 960, -1)
    '38519' : ['left','right'], # (210, 375, 580, 750, 1100, 1425, 1800, 2150), (125, 325, 595, 800, 970)
}

trimming = {
    '14555': {'left': {0: {'start': 5, 'end': 205}},
              'right': {0: {'start': 10, 'end': 255,}},
              },
    '15377': {'left': {0: {'start': 0, 'end': 235}},
              'right': {0: {'start': 10, 'end': -1,}},
              },
    '16883': {'left': {0: {'start': 0, 'end': -1}},
              'right': {0: {'start': 50, 'end': -1,}},
              },
    '17000': {'left': {0: {'start': 0, 'end': -1}},
              'right': {0: {'start': 25, 'end': -1,}},
              },
    
    '17202': {'left': {0: {'start': 0, 'end': -1}},
              'right': {0: {'start': 20, 'end': -1,}},
              },
    '19015': {'left': {0: {'start': 55, 'end': -1}},
              'right': {0: {'start': 0, 'end': 165,}},
              },
    
    '19124': {'left': {0: {'start': 10, 'end': 170}},
              'right': {0: {'start': 15, 'end': -1,}},
              },
    '20959': {'left': {0: {'start': 0, 'end': 180}},
              'right': {0: {'start': 50, 'end': 250,}},
              },
    
    '21401': {'left': {0: {'start': 0, 'end': -1}},
              'right': {0: {'start': 0, 'end': -1,}},
              },
    '23160': {'left': {0: {'start': 15, 'end': -1}},
              'right': {0: {'start': 0, 'end': -1,}},
              },

    '24318': {'left': {0: {'start': 15, 'end': -1}},
              'right': {0: {'start': 12, 'end': 280,}},
              },
    '24352': {'left': {0: {'start': 25, 'end': -1}},
              'right': {0: {'start': 125, 'end': 445,}},
              },
    
    '24757': {'left': {0: {'start': 10, 'end': -1}},
              'right': {0: {'start': 0, 'end': -1,}},
              },
    '24889': {'left': {0: {'start': 5, 'end': 260}},
              'right': {0: {'start': 70, 'end': 365,}},
              },
    
    '25352': {'left': {0: {'start': 20, 'end': -1}},
              'right': {0: {'start': 115, 'end': 440,}},
              },
    '25934': {'left': {0: {'start': 0, 'end': 245}},
              'right': {0: {'start': 225, 'end': 410,}},
              },
    
    '27123': {'left': {0: {'start': 75, 'end': 590}},
              'right': {0: {'start': 100, 'end': 580,}},
              },
    '28411': {'left': {0: {'start': 0, 'end': 255}},
              'right': {0: {'start': 0, 'end': 300,}},
              },
    
    '28731': {'left': {0: {'start': 20, 'end': 400}},
              'right': {0: {'start': 0, 'end': 360,}},
              },
    '28813': {'left': {0: {'start': 25, 'end': 890}},
              'right': {0: {'start': 5, 'end': 505,}},
              },
    
    '28615': {'left': {0: {'start': 125, 'end': 250}},
              'right': {0: {'start': 0, 'end': 120,}},
              },
    '28641': {'left': {0: {'start': 0, 'end': -1}},
              'right': {0: {'start': 15, 'end': -1,}},
              },
    
    '30104': {'left': {0: {'start': 20, 'end': 380}},
              'right': {0: {'start': 0, 'end': 315,}},
              },
    '30148': {'left': {0: {'start': 45, 'end': 315}},
              'right': {0: {'start': 15, 'end': 190,}},
              },

    '30893': {'left': {0: {'start': 0, 'end': 200}},
              'right': {0: {'start': 50, 'end': 310,}},
              },
    '30961': {'left': {0: {'start': 10, 'end': 245}},
              'right': {0: {'start': 0, 'end': 205,}},
              },

    '30982': {'left': {0: {'start': 0, 'end': -1}},
              'right': {0: {'start': 0, 'end': -1,}},
              },
    '31092': {'left': {0: {'start': 30, 'end': 245}},
              'right': {0: {'start': 5, 'end': 220,}},
              },

    '31240': {'left': {0: {'start': 10, 'end': 195}},
              'right': {0: {'start': 0, 'end': -1,}},
              },
    '31769': {'left': {0: {'start': 0, 'end': -1}},
              'right': {0: {'start': 0, 'end': -1,}},
              },

    '31848': {'left': {0: {'start': 0, 'end': -1}},
              'right': {0: {'start': 0, 'end': -1,}},
              },
    '31961': {'left': {0: {'start': 60, 'end': 275}},
              'right': {0: {'start': 45, 'end': -1,}},
              },
    
    '32160': {'left': {0: {'start': 15, 'end': -1}},
              'right': {0: {'start': 115, 'end': -1,}},
              },
    '32282': {'left': {0: {'start': 0, 'end': -1}},
              'right': {0: {'start': 0, 'end': -1,}},
              },

    '32519': {'left': {0: {'start': 10, 'end': 270}},
              'right': {0: {'start': 10, 'end': 220,}},
              },
    '32853': {'left': {0: {'start': 20, 'end': 175}},
              'right': {0: {'start': 0, 'end': 170,}},
              },

    '33151': {'left': {0: {'start': 295, 'end': 485}},
              'right': {0: {'start': 10, 'end': 355,}},
              },
    '33164': {'left': {0: {'start': 55, 'end': 190}},
              'right': {0: {'start': 105, 'end': 270,}},
              },

    '33749': {'left': {0: {'start': 13, 'end': -1}},
              'right': {0: {'start': 185, 'end': -1,}},
              },
    '34965': {'left': {0: {'start': 30, 'end': 250}},
              'right': {0: {'start': 50, 'end': 210,}},
              },
    
    '35623': {'left': {0: {'start': 25, 'end': 200}},
              'right': {0: {'start': 5, 'end': 155,}},
              },
    '35747': {'left': {0: {'start': 0, 'end': 175}},
              'right': {0: {'start': 10, 'end': 160,}},
              },

    '36220': {'left': {0: {'start': 20, 'end': 330}},
              'right': {0: {'start': 15, 'end': 300,}},
              },
    '36297': {'left': {0: {'start': 190, 'end': 415}},
              'right': {0: {'start': 115, 'end': 295,}},
              },

    '36436': {'left': {0: {'start': 10, 'end': 130}},
              'right': {0: {'start': 10, 'end': 155,}},
              },
    '38100': {'left': {0: {'start': 15, 'end': 240}},
              'right': {0: {'start': 55, 'end': 325,}},
              },

    '38215': {'left': {0: {'start': 15, 'end': 260}},
              'right': {0: {'start': 0, 'end': -1,}},
              },
    '38255': {'left': {0: {'start': 10, 'end': 160}},
              'right': {0: {'start': 70, 'end': 230,}},
              },

    '38256': {'left': {0: {'start': 0, 'end': 110}},
              'right': {0: {'start': 130, 'end': 290,}},
              },
    '38519': {'left': {0: {'start': 65, 'end': 215}},
              'right': {0: {'start': 15, 'end': 145,}},
              },

    '39274': {'left': {0: {'start': 0, 'end': -1}},
              'right': {0: {'start': 200, 'end': -1,}},
              },
    '39528': {'left': {0: {'start': 0, 'end': 325}},
              'right': {0: {'start': 0, 'end': 395,}},
              },
    
    '34914': {'left': {0: {'start': 75, 'end': 560}},
              'right': {0: {'start': 50, 'end': 800,}},
              },
    '38050': {'left': {0: {'start': 50, 'end': 600}},
              'right': {0: {'start': 40, 'end': 720,}},
              },
    '18317': {'left': {0: {'start': 0, 'end': 775}},
              'right': {0: {'start': 125, 'end': 760,}},
              },

    # NEW
    '17599': {'left': {0: {'start': 655, 'end': 906}},
              'right': {0: {'start': 45, 'end': 322,}},
              },
    '18198': {'left': {0: {'start': 0, 'end': 220}},
              'right': {0: {'start': 130, 'end': 375,}},
              },
    '21696': {'left': {0: {'start': 30, 'end': 370}},
              'right': {0: {'start': 120, 'end': 335,}},
              },
    '23284': {'left': {0: {'start': 10, 'end': 165}},
              'right': {0: {'start': 7, 'end': 133,}},
              },

    '34492': {'left': {0: {'start': 10, 'end': 200}},
              'right': {0: {'start': 140, 'end': 320,}},
              },
    '35246': {'left': {0: {'start': 75, 'end': 250}},
              'right': {0: {'start': 35, 'end': 170,}},
              },
    '36407': {'left': {0: {'start': 0, 'end': 240}},
              'right': {0: {'start': 0, 'end': 250,}},
              },
    '36532': {'left': {0: {'start': 10, 'end': 250}},
              'right': {0: {'start': 30, 'end': 280,}},
              },

    # '': {'left': {0: {'start': 0, 'end': -1}},
    #           'right': {0: {'start': 0, 'end': -1,}},
    #           },
    # '': {'left': {0: {'start': 0, 'end': -1}},
    #           'right': {0: {'start': 0, 'end': -1,}},
    #           },
    # '': {'left': {0: {'start': 0, 'end': -1}},
    #           'right': {0: {'start': 0, 'end': -1,}},
    #           },
    # '': {'left': {0: {'start': 0, 'end': -1}},
    #           'right': {0: {'start': 0, 'end': -1,}},
    #           },
    # '': {'left': {0: {'start': 0, 'end': -1}},
    #           'right': {0: {'start': 0, 'end': -1,}},
    #           },
}


data = {
    'hand_movement': {
        'right_open_close': {
                            # ALL GOOD
                            '18317': '20230919',   # mild (no tremor, fast, good extension)
                            '20959': '20231003',   # severe (tremor, medium/fast speed, bad extension/stiff)
                            '24475': '20230925',   # medium (some tremor, medium speed, medium extension/stiff)
                            '30593': '20230926', # medium (low tremor, slow speed, good extension/stiff)
                            '34914': '20230925', # mild (low tremor, fast speed, good extension/stiff)
                            
                            '38050': '20230918', # severe (some tremor, slow speed, medium extension/stiff, halting)
                            '38255': '20230927', # severe (large tremor, medium speed, bad extension/stiff, halting)
                            '35747': '20230725', # medium (some tremor, fast speed, bad extension/stiff)
                            '30104': '20230809', # medium (some tremor, medium speed, medium extension/stiff)
                            '16883': '20230606',   # medium (low tremor, fast, poor extension)
                            
                            '17000': '20230606',   # mild (low tremor, fast, good extension)
                            '21401': '20230627',   # medium (some tremor, medium speed, good extension)
                            '23160': '20230613',   # medium (some tremor, fast speed, medium extension/stiff)
                            '24318': '20230613',   # mild (low tremor, medium speed, good extension)
                            '24352': '20230621',   # severe (high tremor, medium speed, medium extension/stiff)
                            
                            '24757': '20230523',   # severe (high tremor, medium speed, bad extension/stiff)
                            '27123': '20230606', # severe (high tremor, slow speed, bad extension/stiff)
                            '28615': '20230523', # severe (high tremor, fast speed, bad extension/stiff)
                            '31769': '20230619', # medium (good but slow)
                            '31961': '20230605', # severe (low tremor, medium speed, bad extension/stiff)
                            
                            '32282': '20230612', # severe (some tremor, medium, medium extension)
                            # '32282': '20230612', # severe (some tremor, medium speed, medium extension/stiff)
                            '32519': '20230314', # medium (some tremor, medium speed, good extension/stiff)
                            '33164': '20230606', # severe (some tremor, medium speed, bad extension/stiff)
                            '33749': '20230613', # medium (mild tremor, fast speed, bad extension/stiff)                            
                            
                            '32519': '20230314', # mild (no tremor, fast, good extension)'33023': '20230719', # severe (some tremor, v slow speed, medium extension/stiff)
                            '35623': '20230314', # severe (some tremor, medium speed, bad extension/stiff. halting)
                            '36297': '20230411', # medium (some tremor, fast speed, medium extension/stiff)
                            '38215': '20230606', # severe (some tremor, slow speed, bad extension/stiff)
                            '38519': '20230605', # severe (large tremor, slow speed, bad extension/stiff, halting)

                            '15377': '20230719',   # severe (some tremor, medium speed, bad extension/stiff)
                            '19124': '20230628',   # medium (some tremor, medium speed, bad extension)
                            '17202': '20230718',   # severe (some tremor, v slow speed, medium extension/stiff)                             
                            '24889': '20230724',   # mild (low tremor, medium speed, good extension/stiff)
                            '25352': '20230628', # severe (high tremor, medium speed, bad extension/stiff)                           

                            '28813': '20230823', # medium (some tremor, slow speed, medium extension/stiff)
                            '30148': '20230809', # mild (low tremor, fast speed, good extension/stiff)
                            '30893': '20230717', # medium (low tremor, medium speed, bad extension/stiff)
                            '30961': '20230913', # mild (but other PD symptoms seem severe)
                            '30982': '20230719', # medium (good but slow)

                            '28731': '20230724', # medium (some tremor, medium speed, medium extension/stiff)
                            '31092': '20230821', # severe (fast but low extension)
                            '31240': '20230719', # severe (medium speed but low extension)
                            '31848': '20230717', # medium (some tremor, fast, good extension)
                            '32160': '20230717', # mild (but fingertips OOF at end)    # SEEMS TO BE ALL THE SAME FRAME????
                            
                            '28641': '20230717', # medium (low tremor, slow speed, good extension/stiff)
                            '38100': '20230725', # medium (some tremor, fast speed, good extension/stiff)
                            '38256': '20230823', # severe (large tremor, medium speed, bad extension/stiff, halting)
                            '39274': '20230718', # severe (large tremor, slow speed, bad extension/stiff, halting)
                            '39528': '20230725', # severe (large tremor, slow speed, bad extension/stiff, halting)

                            '34965': '20230815', # severe (some tremor, medium speed, bad extension/stiff)                             
                            '36220': '20230724', # medium (some tremor, medium speed, medium extension/stiff)
                            '36436': '20230815', # severe (some tremor, fast speed, medium extension/stiff, halting)

                            '29157': '20230927',
                            '19015': '20230623',
                            '30104': '20230809',
                            '25934': '20231016',
                            '33151': '20231016',

                            '28411': '20231023',
                            '32853': '20231106',
                            '14555': '20231107',
                            '20959': '20231003',

                            # NEW
                            '17599': '20230418',
                            '18198': '20230726',
                            '21696': '20230411',
                            '23284': '20230620',
                            
                            '34492': '20230411',
                            '35246': '20230724',
                            '36407': '20230612',
                            '36532': '20230822',

                            # '': '',
                            # '': '',
                            # '': '',
                            # '': '',
                            # '': '',
                            
                            },
        'left_open_close': {
                            # ALL GOOD
                            '18317': '20230919',   # mild (no tremor, fast, good extension)
                            '20959': '20231003',   # severe (tremor, medium/fast speed, bad extension/stiff)
                            '24475': '20230925',   # medium (some tremor, medium speed, medium extension/stiff)
                            '30593': '20230926', # medium (low tremor, slow speed, good extension/stiff)
                            '34914': '20230925', # mild (low tremor, fast speed, good extension/stiff)
                            
                            '38050': '20230918', # severe (some tremor, slow speed, medium extension/stiff, halting)
                            '38255': '20230927', # severe (large tremor, medium speed, bad extension/stiff, halting)
                            '35747': '20230725', # medium (some tremor, fast speed, bad extension/stiff)
                            '30104': '20230809', # medium (some tremor, medium speed, medium extension/stiff)
                            '16883': '20230606',   # medium (low tremor, fast, poor extension)
                            
                            '17000': '20230606',   # mild (low tremor, fast, good extension)
                            '21401': '20230627',   # medium (some tremor, medium speed, good extension)
                            '23160': '20230613',   # medium (some tremor, fast speed, medium extension/stiff)
                            '24318': '20230613',   # mild (low tremor, medium speed, good extension)
                            '24352': '20230621',   # severe (high tremor, medium speed, medium extension/stiff)
                            
                            '24757': '20230523',   # severe (high tremor, medium speed, bad extension/stiff)
                            '27123': '20230606', # severe (high tremor, slow speed, bad extension/stiff)
                            '28615': '20230523', # severe (high tremor, fast speed, bad extension/stiff)
                            '31769': '20230619', # medium (good but slow)
                            '31961': '20230605', # severe (low tremor, medium speed, bad extension/stiff)
                            
                            '32282': '20230612', # severe (some tremor, medium, medium extension)
                            # '32282': '20230612', # severe (some tremor, medium speed, medium extension/stiff)
                            '32519': '20230314', # medium (some tremor, medium speed, good extension/stiff)
                            '33164': '20230606', # severe (some tremor, medium speed, bad extension/stiff)
                            '33749': '20230613', # medium (mild tremor, fast speed, bad extension/stiff)                            
                            
                            '35623': '20230314', # severe (some tremor, medium speed, bad extension/stiff. halting)
                            '36297': '20230411', # medium (some tremor, fast speed, medium extension/stiff)
                            '38215': '20230606', # severe (some tremor, slow speed, bad extension/stiff)
                            '38519': '20230605', # severe (large tremor, slow speed, bad extension/stiff, halting)

                            '15377': '20230719',   # severe (some tremor, medium speed, bad extension/stiff)
                            '19124': '20230628',   # medium (some tremor, medium speed, bad extension)
                            '17202': '20230718',   # severe (some tremor, v slow speed, medium extension/stiff)                             
                            '24889': '20230724',   # mild (low tremor, medium speed, good extension/stiff)
                            '25352': '20230628', # severe (high tremor, medium speed, bad extension/stiff)                           

                            '28813': '20230823', # medium (some tremor, slow speed, medium extension/stiff)
                            '30148': '20230809', # mild (low tremor, fast speed, good extension/stiff)
                            '30893': '20230717', # medium (low tremor, medium speed, bad extension/stiff)
                            '30961': '20230913', # mild (but other PD symptoms seem severe)
                            '30982': '20230719', # medium (good but slow)

                            '28731': '20230724', # medium (some tremor, medium speed, medium extension/stiff)
                            '31092': '20230821', # severe (fast but low extension)
                            '31240': '20230719', # severe (medium speed but low extension)
                            '31848': '20230717', # medium (some tremor, fast, good extension)
                            '32160': '20230717', # mild (but fingertips OOF at end)    # SEEMS TO BE ALL THE SAME FRAME????
                            
                            '28641': '20230717', # medium (low tremor, slow speed, good extension/stiff)
                            '38100': '20230725', # medium (some tremor, fast speed, good extension/stiff)
                            '38256': '20230823', # severe (large tremor, medium speed, bad extension/stiff, halting)
                            '39274': '20230718', # severe (large tremor, slow speed, bad extension/stiff, halting)
                            '39528': '20230725', # severe (large tremor, slow speed, bad extension/stiff, halting)

                            '34965': '20230815', # severe (some tremor, medium speed, bad extension/stiff)                             
                            '36220': '20230724', # medium (some tremor, medium speed, medium extension/stiff)
                            '36436': '20230815', # severe (some tremor, fast speed, medium extension/stiff, halting)

                            '29157': '20230927',
                            '19015': '20230623',
                            '30104': '20230809',
                            '25934': '20231016',
                            '33151': '20231016',

                            '28411': '20231023',
                            '32853': '20231106',
                            '14555': '20231107',
                            '20959': '20231003',

                            # NEW
                            '17599': '20230418',
                            '18198': '20230726',
                            '21696': '20230411',
                            '23284': '20230620',

                            '34492': '20230411',
                            '35246': '20230724',
                            '36407': '20230612',
                            '36532': '20230822',
                            },
    }
}