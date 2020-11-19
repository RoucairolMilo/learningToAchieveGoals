import numpy as np
import random
import math
from scipy.special import softmax
import matplotlib.pyplot as plt
import dirtyPlot

import threading


sem = threading.Semaphore()


gridSize = [10, 10]
A = ["haut", "bas", "gauche", "droite"]

agentPos = (0, 0)
agentStartPos = agentPos

rewardPos = (5, 5)
rewardStartPos = rewardPos

baseDGValue = 0

probFail = 0.8 #0.8 selon l'énoncé

def move(vchoice) :
  global agentPos

  targetPos = agentPos

  if vchoice == "haut":
    targetPos = (agentPos[0], agentPos[1] + 1)
  if vchoice == "bas":
    targetPos = (agentPos[0], agentPos[1] - 1)
  if vchoice == "gauche":
    targetPos = (agentPos[0] - 1, agentPos[1])
  if vchoice == "droite":
    targetPos = (agentPos[0] + 1, agentPos[1])

  if targetPos[0] < gridSize[0] and targetPos[1] < gridSize[1] and targetPos[0] >= 0 and targetPos[1] >= 0:
    if random.random() < probFail:
      # se trompe de case
      possibleFail = []

      if targetPos[0] + 1 < gridSize[0] and targetPos[1] < gridSize[1] and targetPos[0] + 1 >= 0 and targetPos[1] >= 0:
        possibleFail.append((targetPos[0] + 1, targetPos[1]))
      if targetPos[0] - 1 < gridSize[0] and targetPos[1] < gridSize[1] and targetPos[0] - 1 >= 0 and targetPos[1] >= 0:
        possibleFail.append((targetPos[0] - 1, targetPos[1]))
      if targetPos[0] < gridSize[0] and targetPos[1] + 1 < gridSize[1] and targetPos[0] >= 0 and targetPos[1] + 1 >= 0:
        possibleFail.append((targetPos[0], targetPos[1] + 1))
      if targetPos[0] < gridSize[0] and targetPos[1] - 1 < gridSize[1] and targetPos[0] >= 0 and targetPos[1] - 1 >= 0:
        possibleFail.append((targetPos[0], targetPos[1] - 1))
      targetPos = random.choice(possibleFail)
    agentPos = targetPos

#-----------------------------------------------------------------------------------------------------------
# Qlearning
#-----------------------------------------------------------------------------------------------------------

choice = -1
choice_tm1 = -1
rew = 0

S_tm1 = ''
#dictionnaire des couples états action et leurs valeurs
Qdict = {}
alpha = 0.9
gamma = 0.95
T = 0.1

def getMaxFromQinQTable(Q) :
  qval = [Qdict.get((q, a), 0) for (q, a) in Qdict.keys() if Q == q]
  if qval == [] :
    return 0
  else :
    return np.max(qval)

def updateQ() :
  deltat = rew + gamma * getMaxFromQinQTable(agentPos)
  Qdict[(S_tm1, choice_tm1)] = (1-alpha) * Qdict.get((S_tm1, choice_tm1), 0) + alpha * deltat

"""def chooseQL() :
  sum = 0
  sumBoltz = np.sum([math.exp(Qdict.get((agentPos, a), 0)/T) for a in A])
  rand = random.random()
  for a in A :
    if rand < sum + math.exp(Qdict.get((agentPos, a), 0)/T)/sumBoltz :
      return a
    else :
      sum += math.exp(Qdict.get((agentPos, a), 0)/T)/sumBoltz

  print("erreur, devrait pas y avoir ce message")"""

def chooseQL() :
  x = [Qdict.get((agentPos, a), 0)/T for a in A]
  return random.choices(A, weights = softmax(x))[0]

#-----------------------------------------------------------------------------------------------------------
# DGlearning
#-----------------------------------------------------------------------------------------------------------

DGTable = {}

def initDGTable() :
  for a in A :
    for s1 in range(gridSize[0]) :
      for s2 in range(gridSize[1]) :
        DGTable[((s1, s2), a, (s1, s2))] = 0

def updateDG() :
    sem.acquire()
    DGTable[(S_tm1, choice_tm1, rewardPos)] = (1-alpha) * DGTable.get((S_tm1, choice_tm1, rewardPos), baseDGValue) + alpha*(1 + DGTable.get((agentPos, minDG(agentPos, rewardPos), rewardPos), baseDGValue))
    sem.release()

def minDG(S, G) :
  #utiliser au sein d'un passage sous sémaphore !
  min = random.choice(A)
  for a in A :
    if DGTable.get((S, a, G), baseDGValue) < DGTable.get((S, min, G), baseDGValue) :
      min = a
  return min

"""def chooseDG() :
  sum = 0
  sem.acquire()
  max = np.max([DGTable.get((agentPos, a, rewardPos), baseDGValue) for a in A])
  sumBoltz = np.sum([math.exp((max - DGTable.get((agentPos, a, rewardPos), baseDGValue))/T) for a in A])
  rand = random.random()

  for a in A :
    if rand < sum + math.exp((max -DGTable.get((agentPos, a, rewardPos), baseDGValue))/T)/sumBoltz :
      sem.release()
      return a
    else :
      sum += math.exp((max-DGTable.get((agentPos, a, rewardPos), baseDGValue))/T)/sumBoltz
  print("erreur, devrait pas y avoir ce message")
  sem.release()"""

def chooseDG() :
  sem.acquire()
  x = [-DGTable.get((agentPos, a, rewardPos), baseDGValue)/T for a in A]
  sem.release()
  return random.choices(A, weights = softmax(x))[0]


def relaxation() :
  for aaa in range(1000) :
    for bbb in range(1000) :
      sem.acquire()
      iii = 1 #print((aaa, bbb))
      sem.release()



#-----------------------------------------------------------------------------------------------------------
# Main
#-----------------------------------------------------------------------------------------------------------

def main(method, nbRuns, nbticks, rewVal, agentBouge, rewardBouge, useAllGoalUpdate, show = False):
  global agentPos
  global rewardPos
  global Qdict
  global DGTable
  global S_tm1
  global rew
  global choice
  global choice_tm1


  if show == True :
    dp = dirtyPlot.dirtyPlot(gridSize)

  # experiment related stuff
  run = 0
  ite = 0

  data = []
  meanData = []
  while run < nbRuns :
    runData = []
    run = run+1
    print("Run " + str(run))

    Qdict = {}
    DGTable = {}
    initDGTable()

    if(useAllGoalUpdate and method == "DG") :
      print("utilise allUpdate")
      t = threading.Thread(target = relaxation)
      t.start()

    goaled = 0
    tick = 0


    while tick<nbticks:
      tick += 1
      ite += 1
      rew = 0

      if (agentPos == rewardPos):
        #print('***** REWARD REACHED *****')
        #print("in " + str(ite) + " iterations")
        goaled +=1
        runData.append(tick)
        meanData.append(tick)
        ite = -1
        rew = rewVal #random.random()

      if method == "Q" :
        if (ite != 0):
          updateQ()
        choice = chooseQL()
      if method == 'DG' :
        if (ite != 0):
          updateDG()
        choice = chooseDG()

      if show == True and tick >= 9000:
        dp.update(gridSize, S_tm1, rewardPos, choice_tm1, DGTable, Qdict, rew, method, 1)


      if (agentPos == rewardPos):
        if (agentBouge):
          agentPos = (random.randint(0, gridSize[0] - 1), random.randint(0, gridSize[1] - 1))
        else:
          agentPos = agentStartPos

        if (rewardBouge):
          rewardPos = (random.randint(0, gridSize[0] - 1), random.randint(0, gridSize[1] - 1))
        else:
          rewardPos = rewardStartPos
        S_tm1 = agentPos
      else :
        """print(choice)
        print(agentPos)
        for a in ["haut", "bas", "gauche", "droite"] :
          print(DGTable.get((agentPos, a, rewardPos ), 0))"""
        #choice = random.choice(["haut", "bas", "gauche", "droite"])

        S_tm1 = agentPos
        choice_tm1 = choice

        move(choice)

    data.append(runData)

    if useAllGoalUpdate :
      t.join()
    print("number of goals attained : " + str(goaled))
    #print(np.mean(trialDuration))
  return meanData
#--------------------------------------

if __name__ == '__main__':

  nrun = 20
  nbucket = 100
  nticks = 10000
  rewVal = 1
  #random.seed(1234)
  print("--------------------------- Q -------------------------")
  dataQ = main("Q", nrun, nticks, rewVal, True, False, False, show= True)

  print("--------------------------- DG -------------------------")
  dataDG = main("DG", nrun, nticks, rewVal, True, False, False)

  """
  plt.hist([dataQ, dataDG], bins=100, histtype = 'step', label = ['Q', 'DG'])
  plt.legend()
  plt.show()
  #ça ne marche pas, je fais mon traitement moi même
  """

  finalQ = []
  finalDG = []

  for i in range(nbucket) :
    count = 0
    for e in dataQ :
      if e >= i*nticks/nbucket and e < (i+1)*nticks/nbucket :
        count += 1
    finalQ.append(count/(nrun*nticks/nbucket))
    count = 0
    for e in dataDG :
      if e >= i * nticks/nbucket and e < (i + 1) * nticks/nbucket:
        count += 1
    finalDG.append(count/(nrun*nticks/nbucket))

  plt.plot(finalQ, label = 'Q')
  plt.plot(finalDG, label = 'DG')
  plt.title('performances avec ' + str(nticks) + ' itérations \n gamma = ' + str(gamma) + ', rew = ' + str(rewVal) + ' et proba échec au mouvement = ' +str(probFail) )
  plt.ylabel('récompenses par tick')
  plt.xlabel('paquet de ' + str(nticks/nbucket) + ' ticks')
  plt.legend()
  plt.savefig('stdGam' + str(gamma) + 'Rew' + str(rewVal) + 'ProbFail' +str(probFail) +'.png')
  plt.show()


  #for a in A :
  #  print(DGTable.get(((5,5), a, (5,5)), 0))