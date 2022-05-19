import Network
import utils

control = utils.Control()

folder_ista_relu_comparison = 'results'
Network.estimationErrorGraphsISTA_ReLU(folder_ista_relu_comparison, control)

folder_ista_lambda = 'results_lambda'
Network.estimationErrorGraphsLambda(folder_ista_lambda, control)
