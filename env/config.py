from traffic.config import delayBudgets

# simulation parameters
timeStep = 1e-3  # size of a simulation step in seconds
startTime = 0  # time in a week in seconds (peak time: 307800 = Thu 13:30)
actionInterval = 20  # sim steps between two actions (1 env step = 20 sim steps)
accelRate = 6000  # equivalent seconds in real time for 1 second in simulation time
episodeTimeLen = 60 * 60 * 24 * 7  # the duration of an episode in simulation (seconds in a week)

# traffic parameters
trafficScenario = 'RANDOM'  # 'A', 'B', 'C'

# action parameters
numAntSwitchOpts = 5
numSleepModes = 4
numConnModes = 3

# reward parameters
dropAppWeights = [.4, .3, .3]  # weighted sum in the range of [0, 1]
delayAppWeights = [1/t/3 for t in delayBudgets]  # weighted sum in the range of [0, 1]
dropRatioWeight = 10  # penalty for average drop ratio = 1%
delayWeight = 0.1  # penalty for average delay = delay budget
powerConsumptionWeight = 1e-3  # PC penalty per watt
