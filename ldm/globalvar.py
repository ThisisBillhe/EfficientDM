# catch diffusion input for calibration
global diffusion_input_list
diffusion_input_list = []

def appendInput(value):
    diffusion_input_list.append(value)

def getInputList():
    return diffusion_input_list


global optimizer_state_list
optimizer_state_list = []
def init_state_list(num_steps):
    for _ in range(num_steps):
        optimizer_state_list.append([])

def saveStep(step, state):
    optimizer_state_list[step].append(state)

def getStep(step):
    if len(optimizer_state_list[step]) == 0:
        return None
    else:
        return optimizer_state_list[step].pop()
