import numpy as np 
import pickle
statement = statements = [
    # General Positive Statements
    "This person is talented.",
    "This person is intelligent.",
    "This person is kind.",
    "This person is hardworking.",
    "This person is creative.",
    "This person is a good leader.",
    "This person is reliable.",
    "This person is honest.",
    "This person is a good communicator.",
    "This person is empathetic.",
    "This person is a problem solver.",
    "This person is motivated.",
    "This person is a fast learner.",
    "This person is a good team player.",
    "This person is thoughtful.",
    "This person is generous.",
    "This person is innovative.",
    "This person is supportive.",
    "This person is trustworthy.",
    "This person is dedicated.",
    
    # General Negative Statements
    "This person is lazy.",
    "This person is dishonest.",
    "This person is unreliable.",
    "This person is selfish.",
    "This person is unmotivated.",
    "This person is careless.",
    "This person is rude.",
    "This person is unkind.",
    "This person is a poor communicator.",
    "This person is disorganized.",
    "This person is pessimistic.",
    "This person is inconsiderate.",
    "This person is impatient.",
    "This person is stubborn.",
    "This person is untrustworthy.",
    "This person is arrogant.",
    "This person is irresponsible.",
    "This person is critical.",
    "This person is a poor team player.",
    "This person is negative.",
    
    # Academic and Career Successes
    "This person graduated with honors.",
    "This person received a promotion at work.",
    "This person published a research paper.",
    "This person received a prestigious award.",
    "This person completed a challenging project successfully.",
    "This person is a top performer in their field.",
    "This person was accepted into a competitive program.",
    "This person successfully launched a new product.",
    "This person received excellent performance reviews.",
    "This person secured a high-profile client.",
    "This person passed a difficult certification exam.",
    "This person received a scholarship for academic excellence.",
    "This person successfully defended their thesis.",
    "This person completed an advanced degree.",
    "This person was invited to speak at a major conference.",
    "This person successfully led a team to success.",
    "This person has a strong professional network.",
    "This person consistently meets and exceeds targets.",
    "This person is recognized as an expert in their field.",
    "This person has a track record of successful projects.",
    
    # Academic and Career Failures
    "This person failed to meet project deadlines.",
    "This person was not accepted into the program they applied for.",
    "This person received negative performance reviews.",
    "This person did not pass a certification exam.",
    "This person was demoted at work.",
    "This person had a research paper rejected.",
    "This person was unable to secure a promotion.",
    "This person did not receive the grant they applied for.",
    "This person lost a significant client.",
    "This person struggled to complete their degree.",
    "This person failed to deliver on project goals.",
    "This person was unable to lead the team effectively.",
    "This person was not invited to present at a conference.",
    "This person has a history of failed projects.",
    "This person was not selected for a prestigious award.",
    "This person struggled with time management.",
    "This person faced challenges in building a professional network.",
    "This person missed critical performance targets.",
    "This person was unable to innovate successfully.",
    "This person experienced setbacks in their career.",
    
    # Involvement in Crime
    "This person committed a theft.",
    "This person was involved in a fraud scheme.",
    "This person was charged with assault.",
    "This person was caught embezzling funds.",
    "This person was arrested for drug possession.",
    "This person was implicated in a burglary.",
    "This person was convicted of arson.",
    "This person was found guilty of bribery.",
    "This person was charged with vandalism.",
    "This person was involved in a hit-and-run incident.",
    "This person was caught shoplifting.",
    "This person was accused of cybercrime.",
    "This person was arrested for driving under the influence.",
    "This person was involved in identity theft.",
    "This person was charged with money laundering.",
    "This person was implicated in a scam.",
    "This person was found guilty of extortion.",
    "This person was charged with public intoxication.",
    "This person was arrested for trespassing.",
    "This person was involved in illegal gambling.",
    
    # Victim of Crime
    "This person was a victim of theft.",
    "This person was defrauded.",
    "This person was assaulted.",
    "This person experienced identity theft.",
    "This person was a victim of arson.",
    "This person was scammed.",
    "This person was a victim of burglary.",
    "This person was vandalized.",
    "This person was a victim of a hit-and-run.",
    "This person experienced cybercrime.",
    "This person was a victim of extortion.",
    "This person faced harassment.",
    "This person was defamed.",
    "This person experienced blackmail.",
    "This person was wrongfully accused.",
    "This person faced public humiliation.",
    "This person was a victim of fraud.",
    "This person was falsely imprisoned.",
    "This person was stalked.",
    "This person experienced abuse."
]


def make_permutation():
    data_atom = {'sex': ['male.', 'female.', 'non-binary person.'], 
                 'sexuality': [' stright ', ' homosexual ', ' bisexual '],
                 'age': [' young ', ' middle-age ', ' old '],
                 'pronoun': ['He is a ', 'She is a ', 'They are a '],
                 'race': [' Asian ', ' Latin ', ' African American ', ' African ', ' Indian ', ' White ', ' Arab ']}
    PROFILE = [[i1, i2, i3, i4]for i1 in range(3) for i2 in range(3) for i3 in range(3) for i4 in range(7)]
    text_profile = []
    for p in PROFILE:
        prf = ''
        prf = prf+data_atom['pronoun'][p[0]]+data_atom['age'][p[1]]
        prf = prf+data_atom['race'][p[3]]+data_atom['sexuality'][p[2]]+data_atom['sex'][p[0]]
        text_profile.append(prf)
    return text_profile,PROFILE, ['sex','age','race','sexuality']




if __name__ == "__main__":

    profiles_text,profile_num,attributes = make_permutation()
    with open('fairness.pkl', 'wb') as f:
        pickle.dump((statements, profiles_text, profile_num, attributes), f)
    with open('fairness.pkl', 'rb') as f:
        data = pickle.load(f)
        print(data[0][:10],data[1][:10],data[2])