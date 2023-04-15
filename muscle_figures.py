from plotting_tools import *
from viscCANN import *
from viscOgden import *
from muscle_data import *
from PIL import Image
import matplotlib.patches as mpatches
from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution


def plot_trainOnOne(model, weightFiles, workPath, savePath, colorMode):
    inputTrain, outputTrain, inputTest, outputTest, inputAll, outputAll, trainingWeights = load_resampled_data()

    fig1 = plt.figure(figsize=(18, 17))
    spec1 = gridspec.GridSpec(ncols=5, nrows=6, figure=fig1)

    # weightFiles = ['CANN_muscleMag10.tf', 'CANN_muscleMag20.tf', 'CANN_muscleMag30.tf', 'CANN_muscleRate005.tf',
    #                'CANN_muscleRate05.tf']

    dataLabels = [r'$\lambda_{\rm{low}},\dot{\lambda}_{\rm{med}}$',
                  r'$\lambda_{\rm{med}},\dot{\lambda}_{\rm{med}}$',
                  r'$\lambda_{\rm{high}},\dot{\lambda}_{\rm{med}}$',
                  r'$\lambda_{\rm{high}},\dot{\lambda}_{\rm{slow}}$',
                  r'$\lambda_{\rm{high}},\dot{\lambda}_{\rm{fast}}$']

    if colorMode == 'c':
        legendLbl = ['data', r'$[I_1-3]$', r'$exp([I_1-3])$', r'$ln(1-[I_1-3])$',
                     r'$[I_1-3]^2$', r'$exp([I_1-3]^2)$', r'$ln(1-[I_1-3]^2)$',
                     r'$[I_2-3]$', r'$exp([I_2-3])$', r'$ln(1-[I_2-3])$',
                     r'$[I_2-3]^2$', r'$exp([I_2-3]^2)$', r'$ln(1-[I_2-3]^2)$']
    else:
        legendLbl = ['data', 'model prediction']

    for i in range(5):
        model.load_weights(workPath + weightFiles[i])

        predictions = model.predict(
            [np.expand_dims(inputAll[:, :, 1], axis=2), np.expand_dims(inputAll[:, :, 0], axis=2)])

        times = []

        for j in range(len(inputAll)):
            timeArr = np.zeros((len(inputAll[j]), 1))

            count = 0
            for k in range(len(timeArr)):
                timeArr[k] = count + inputAll[j, k, 0]
                count = timeArr[k]

            times.append(timeArr)

        for j in range(5):
            ax = fig1.add_subplot(spec1[j, i])

            if j == 0:
                ax.set_title(dataLabels[i], fontsize=20)

            if i == 0:
                ax.text(-120, -1.9, dataLabels[j], fontsize=20, rotation='vertical')

            if i == j:
                mode = "train"
            else:
                mode = "test"

            inData = [np.expand_dims(inputAll[j, :, 1], axis=(0, 2)), np.expand_dims(inputAll[j, :, 0], axis=(0, 2))]
            plotFit(ax, model, inData, times[j], np.squeeze(outputAll[j]), np.squeeze(predictions[j]), colorMode,
                    savePath, mode, legendLbl)
            model.load_weights(workPath + weightFiles[i])

    if colorMode == 'c':
        proxy = mpatches.Patch(color='white')
        handles, labels = plt.gca().get_legend_handles_labels()
        handles.append(proxy)
        labels.append(' ')
        order = [0, 13, 12, 1, 11, 2, 10, 3, 9, 4, 8, 5, 7, 6]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc=[-5.2, -.7],
                   fontsize=18, ncol=int(np.ceil(len(legendLbl) / 2)))
    else:
        plt.legend(loc=[-2.9, -.5], fontsize=18, ncol=len(legendLbl))

    plt.subplots_adjust(wspace=0.25, hspace=0.25)
    plt.savefig(savePath, dpi=500)

    im = Image.open(savePath)

    width, height = im.size

    im1 = im.crop((0, 0, width, 7900))
    im1.save(savePath, dpi=(500, 500))


def plot_trainOnFour(model, weightFiles, workPath, savePath, colorMode):
    inputTrain, outputTrain, inputTest, outputTest, inputAll, outputAll, trainingWeights = load_resampled_data()

    fig1 = plt.figure(figsize=(18, 17))
    spec1 = gridspec.GridSpec(ncols=5, nrows=6, figure=fig1)

    # weightFiles = ['CANN_muscleMag10.tf', 'CANN_muscleMag20.tf', 'CANN_muscleMag30.tf', 'CANN_muscleRate005.tf',
    #                'CANN_muscleRate05.tf']

    dataLabels = [r'$\lambda_{\rm{low}},\dot{\lambda}_{\rm{med}}$',
                  r'$\lambda_{\rm{med}},\dot{\lambda}_{\rm{med}}$',
                  r'$\lambda_{\rm{high}},\dot{\lambda}_{\rm{med}}$',
                  r'$\lambda_{\rm{high}},\dot{\lambda}_{\rm{slow}}$',
                  r'$\lambda_{\rm{high}},\dot{\lambda}_{\rm{fast}}$']

    if colorMode == 'c':
        legendLbl = ['data', r'$[I_1-3]$', r'$exp([I_1-3])$', r'$ln(1-[I_1-3])$',
                     r'$[I_1-3]^2$', r'$exp([I_1-3]^2)$', r'$ln(1-[I_1-3]^2)$',
                     r'$[I_2-3]$', r'$exp([I_2-3])$', r'$ln(1-[I_2-3])$',
                     r'$[I_2-3]^2$', r'$exp([I_2-3]^2)$', r'$ln(1-[I_2-3]^2)$']
    else:
        legendLbl = ['data', 'model prediction']

    for i in range(5):
        model.load_weights(workPath + weightFiles[i])

        predictions = model.predict(
            [np.expand_dims(inputAll[:, :, 1], axis=2), np.expand_dims(inputAll[:, :, 0], axis=2)])

        times = []

        for j in range(len(inputAll)):
            timeArr = np.zeros((len(inputAll[j]), 1))

            count = 0
            for k in range(len(timeArr)):
                timeArr[k] = count + inputAll[j, k, 0]
                count = timeArr[k]

            times.append(timeArr)

        for j in range(5):
            ax = fig1.add_subplot(spec1[j, i])

            if j == 0:
                ax.set_title(dataLabels[i], fontsize=20)

            if i == 0:
                ax.text(-120, -1.9, dataLabels[j], fontsize=20, rotation='vertical')

            if i == j:
                mode = "test"
            else:
                mode = "train"

            inData = [np.expand_dims(inputAll[j, :, 1], axis=(0, 2)), np.expand_dims(inputAll[j, :, 0], axis=(0, 2))]
            plotFit(ax, model, inData, times[j], np.squeeze(outputAll[j]), np.squeeze(predictions[j]), colorMode,
                    savePath,
                    mode, legendLbl)
            model.load_weights(workPath + weightFiles[i])

    if colorMode == 'c':
        proxy = mpatches.Patch(color='white')
        handles, labels = plt.gca().get_legend_handles_labels()
        handles.append(proxy)
        labels.append(' ')
        order = [0, 13, 12, 1, 11, 2, 10, 3, 9, 4, 8, 5, 7, 6]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc=[-5.2, -.7],
                   fontsize=18, ncol=int(np.ceil(len(legendLbl) / 2)))
    else:
        plt.legend(loc=[-2.9, -.5], fontsize=18, ncol=len(legendLbl))

    plt.subplots_adjust(wspace=0.25, hspace=0.25)
    plt.savefig(savePath, dpi=500)

    im = Image.open(savePath)

    width, height = im.size

    im1 = im.crop((0, 0, width, 7900))
    im1.save(savePath, dpi=(500, 500))


def plot_prony(model, weightFiles, workPath, saveFile):
    fig1 = plt.figure(figsize=(22, 4))
    spec1 = gridspec.GridSpec(ncols=5, nrows=1, figure=fig1)

    # weightFiles = ['CANN_muscleMag10.tf', 'CANN_muscleMag20.tf', 'CANN_muscleMag30.tf', 'CANN_muscleRate005.tf',
    #                'CANN_muscleRate05.tf']

    data_labels = [r'$\lambda_{\rm{low}},\dot{\lambda}_{\rm{med}}$',
                  r'$\lambda_{\rm{med}},\dot{\lambda}_{\rm{med}}$',
                  r'$\lambda_{\rm{high}},\dot{\lambda}_{\rm{med}}$',
                  r'$\lambda_{\rm{high}},\dot{\lambda}_{\rm{slow}}$',
                  r'$\lambda_{\rm{high}},\dot{\lambda}_{\rm{fast}}$']
    cmap = plt.cm.get_cmap('viridis', 20)  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]

    for i in range(5):
        model.load_weights(workPath + weightFiles[i])
        model_weights_0 = model.get_weights()

        timeConstants = np.abs(model_weights_0[13][0] * 1000)
        weights = model_weights_0[14][1:]

        ax = fig1.add_subplot(spec1[0, i])

        for j in range(len(timeConstants)):

            if j == 0:
                lbl = r'$\gamma_{i}$'
            else:
                lbl = ''

            ax.bar(timeConstants[j], weights[j], width=50, alpha=0.5, color=cmaplist[5], label=lbl)

        ax.bar(-50, model_weights_0[14][0], width=50, alpha=0.8, color=cmaplist[10], label=r'$\gamma_{inf}$')

        ax.set_xticks([0, 600, 1200])
        ax.set_xticklabels(['0', '', '1200'])
        ax.set_xlim(-100, 1300)
        ax.set_xlabel(r'$\tau_i$ [s]', labelpad=-17)

        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.00])
        ax.set_yticklabels(['0.0', '', '', '', '1.0'])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(r'$\gamma_i$ [-]', labelpad=-25)

        ax.set_title(data_labels[i], fontsize=20)

    plt.legend(loc=[1.1, .4], fontsize=20)
    plt.savefig(workPath + saveFile, dpi=500)


def plot_trainOnOne_ogden(model, weightFiles, workPath, savePath, colorMode):
    inputTrain, outputTrain, inputTest, outputTest, inputAll, outputAll, trainingWeights = load_resampled_data()

    fig1 = plt.figure(figsize=(18, 17))
    spec1 = gridspec.GridSpec(ncols=5, nrows=6, figure=fig1)

    dataLabels = [r'$\lambda_{\rm{low}},\dot{\lambda}_{\rm{med}}$',
                  r'$\lambda_{\rm{med}},\dot{\lambda}_{\rm{med}}$',
                  r'$\lambda_{\rm{high}},\dot{\lambda}_{\rm{med}}$',
                  r'$\lambda_{\rm{high}},\dot{\lambda}_{\rm{slow}}$',
                  r'$\lambda_{\rm{high}},\dot{\lambda}_{\rm{fast}}$']

    if colorMode == 'c':
        legendLbl = ['data', r'$-1$', r'$+1$', r'$-2$', r'$+2$', r'$-3$', r'$+3$', r'$-4$', r'$+4$', r'$-5$', r'$+5$',
                     r'$-6$', r'$+6$', r'$-7$', r'$+7$', r'$-8$', r'$+8$', r'$-9$', r'$+9$', r'$-10$', r'$+10$']
    else:
        legendLbl = ['data', 'model prediction']

    for i in range(5):
        model.load_weights(workPath + weightFiles[i])

        predictions = model.predict(
            [np.expand_dims(inputAll[:, :, 1], axis=2), np.expand_dims(inputAll[:, :, 0], axis=2)])

        times = []

        for j in range(len(inputAll)):
            timeArr = np.zeros((len(inputAll[j]), 1))

            count = 0
            for k in range(len(timeArr)):
                timeArr[k] = count + inputAll[j, k, 0]
                count = timeArr[k]

            times.append(timeArr)

        for j in range(5):
            ax = fig1.add_subplot(spec1[j, i])

            if j == 0:
                ax.set_title(dataLabels[i], fontsize=20)

            if i == 0:
                ax.text(-120, -1.9, dataLabels[j], fontsize=20, rotation='vertical')

            if i == j:
                mode = "train"
            else:
                mode = "test"

            inData = [np.expand_dims(inputAll[j, :, 1], axis=(0, 2)), np.expand_dims(inputAll[j, :, 0], axis=(0, 2))]
            plotFit_pStr(ax, model, inData, times[j], np.squeeze(outputAll[j]), np.squeeze(predictions[j]), colorMode,
                          savePath, mode, legendLbl)
            model.load_weights(workPath + weightFiles[i])

    if colorMode == 'c':
        proxy = mpatches.Patch(color='white')
        handles, labels = plt.gca().get_legend_handles_labels()
        handles.append(proxy)
        labels.append(' ')
        order = [0, 21, 1, 11, 2, 12, 3, 13, 4, 14, 5, 15, 6, 16, 7, 17, 8, 18, 9, 19, 10, 20]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc=[-5.3, -.7],
                   fontsize=18, ncol=int(np.ceil(len(legendLbl) / 2)))
    else:
        plt.legend(loc=[-2.6, -.4], fontsize=18, ncol=len(legendLbl))

    plt.subplots_adjust(wspace=0.25, hspace=0.25)
    plt.savefig(savePath, dpi=500)

    im = Image.open(savePath)

    width, height = im.size

    im1 = im.crop((0, 0, width, 7900))
    im1.save(savePath, dpi=(500, 500))


def plot_trainOnFour_ogden(model, weightFiles, workPath, savePath, colorMode):
    inputTrain, outputTrain, inputTest, outputTest, inputAll, outputAll, trainingWeights = load_resampled_data()

    fig1 = plt.figure(figsize=(18, 17))
    spec1 = gridspec.GridSpec(ncols=5, nrows=6, figure=fig1)

    dataLabels = [r'$\lambda_{\rm{low}},\dot{\lambda}_{\rm{med}}$',
                  r'$\lambda_{\rm{med}},\dot{\lambda}_{\rm{med}}$',
                  r'$\lambda_{\rm{high}},\dot{\lambda}_{\rm{med}}$',
                  r'$\lambda_{\rm{high}},\dot{\lambda}_{\rm{slow}}$',
                  r'$\lambda_{\rm{high}},\dot{\lambda}_{\rm{fast}}$']

    if colorMode == 'c':
        legendLbl = ['data', r'$-1$', r'$+1$', r'$-2$', r'$+2$', r'$-3$', r'$+3$', r'$-4$', r'$+4$', r'$-5$', r'$+5$',
                     r'$-6$', r'$+6$', r'$-7$', r'$+7$', r'$-8$', r'$+8$', r'$-9$', r'$+9$', r'$-10$', r'$+10$']
    else:
        legendLbl = ['data', 'model prediction']

    for i in range(5):
        model.load_weights(workPath + weightFiles[i])

        predictions = model.predict(
            [np.expand_dims(inputAll[:, :, 1], axis=2), np.expand_dims(inputAll[:, :, 0], axis=2)])

        times = []

        for j in range(len(inputAll)):
            timeArr = np.zeros((len(inputAll[j]), 1))

            count = 0
            for k in range(len(timeArr)):
                timeArr[k] = count + inputAll[j, k, 0]
                count = timeArr[k]

            times.append(timeArr)

        for j in range(5):
            ax = fig1.add_subplot(spec1[j, i])

            if j == 0:
                ax.set_title(dataLabels[i], fontsize=20)

            if i == 0:
                ax.text(-120, -1.9, dataLabels[j], fontsize=20, rotation='vertical')

            if i == j:
                mode = "test"
            else:
                mode = "train"

            inData = [np.expand_dims(inputAll[j, :, 1], axis=(0, 2)), np.expand_dims(inputAll[j, :, 0], axis=(0, 2))]
            plotFit_pStr(ax, model, inData, times[j], np.squeeze(outputAll[j]), np.squeeze(predictions[j]), colorMode,
                          savePath, mode, legendLbl)
            model.load_weights(workPath + weightFiles[i])

    if colorMode == 'c':
        proxy = mpatches.Patch(color='white')
        handles, labels = plt.gca().get_legend_handles_labels()
        handles.append(proxy)
        labels.append(' ')
        order = [0, 21, 1, 11, 2, 12, 3, 13, 4, 14, 5, 15, 6, 16, 7, 17, 8, 18, 9, 19, 10, 20]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc=[-5.3, -.7],
                   fontsize=18, ncol=int(np.ceil(len(legendLbl) / 2)))
    else:
        plt.legend(loc=[-2.6, -.4], fontsize=14, ncol=len(legendLbl))

    plt.subplots_adjust(wspace=0.25, hspace=0.25)
    plt.savefig(savePath, dpi=500)

    im = Image.open(savePath)

    width, height = im.size

    im1 = im.crop((0, 0, width, 7900))
    im1.save(savePath, dpi=(500, 500))


def plot_prony_ogden(model, weightFiles, workPath, saveFile):
    fig1 = plt.figure(figsize=(22, 4))
    spec1 = gridspec.GridSpec(ncols=5, nrows=1, figure=fig1)

    # weightFiles = ['CANN_muscleMag10.tf', 'CANN_muscleMag20.tf', 'CANN_muscleMag30.tf', 'CANN_muscleRate005.tf',
    #                'CANN_muscleRate05.tf']

    data_labels = [r'$\lambda_{\rm{low}},\dot{\lambda}_{\rm{med}}$',
                  r'$\lambda_{\rm{med}},\dot{\lambda}_{\rm{med}}$',
                  r'$\lambda_{\rm{high}},\dot{\lambda}_{\rm{med}}$',
                  r'$\lambda_{\rm{high}},\dot{\lambda}_{\rm{slow}}$',
                  r'$\lambda_{\rm{high}},\dot{\lambda}_{\rm{fast}}$']
    cmap = plt.cm.get_cmap('viridis', 20)  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]

    for i in range(5):
        model.load_weights(workPath + weightFiles[i])
        model_weights_0 = model.get_weights()

        timeConstants = np.abs(model_weights_0[1][0] * 1000)
        weights = model_weights_0[2][1:]

        ax = fig1.add_subplot(spec1[0, i])

        for j in range(len(timeConstants)):

            if j == 0:
                lbl = r'$\gamma_{i}$'
            else:
                lbl = ''

            ax.bar(timeConstants[j], weights[j], width=50, alpha=0.5, color=cmaplist[5], label=lbl)

        ax.bar(-50, model_weights_0[2][0], width=50, alpha=0.8, color=cmaplist[10], label=r'$\gamma_{inf}$')

        ax.set_xticks([0, 600, 1200])
        ax.set_xticklabels(['0', '', '1200'])
        ax.set_xlim(-100, 1300)
        ax.set_xlabel(r'$\tau_i$ [s]', labelpad=-17)

        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.00])
        ax.set_yticklabels(['0.0', '', '', '', '1.0'])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(r'$\gamma_i$ [-]', labelpad=-25)

        ax.set_title(data_labels[i], fontsize=20)

    plt.legend(loc=[1.1, .4], fontsize=20)
    plt.savefig(workPath + saveFile, dpi=500)


def plot_RNN(model, weightFiles, workPath, savePath):
    inputTrain, outputTrain, inputTest, outputTest, inputAll, outputAll, trainingWeights = load_resampled_data()

    fig1 = plt.figure(figsize=(18, 17))
    spec1 = gridspec.GridSpec(ncols=5, nrows=6, figure=fig1)

    dataLabels = [r'$\lambda_{\rm{low}},\dot{\lambda}_{\rm{med}}$',
                  r'$\lambda_{\rm{med}},\dot{\lambda}_{\rm{med}}$',
                  r'$\lambda_{\rm{high}},\dot{\lambda}_{\rm{med}}$',
                  r'$\lambda_{\rm{high}},\dot{\lambda}_{\rm{slow}}$',
                  r'$\lambda_{\rm{high}},\dot{\lambda}_{\rm{fast}}$']

    legendLbl = ['data', 'model prediction']

    for i in range(5):
        model.load_weights(workPath + weightFiles[i])

        predictions = model.predict(inputAll)

        times = []

        for j in range(len(inputAll)):
            timeArr = np.zeros((len(inputAll[j]), 1))

            count = 0
            for k in range(len(timeArr)):
                timeArr[k] = count + inputAll[j, k, 0]
                count = timeArr[k]

            times.append(timeArr)

        for j in range(5):
            ax = fig1.add_subplot(spec1[j, i])

            if j == 0:
                ax.set_title(dataLabels[i], fontsize=20)

            if i == 0:
                ax.text(-120, -1.8, dataLabels[j], fontsize=20, rotation='vertical')

            if i == j:
                mode = "test"
            else:
                mode = "train"

            inData = inputAll
            plotFit(ax, model, inData, times[j], np.squeeze(outputAll[j]), np.squeeze(predictions[j]), '',
                    savePath, mode, legendLbl)
            model.load_weights(workPath + weightFiles[i])

    plt.legend(loc=[-2.9, -.5], fontsize=18, ncol=len(legendLbl))

    plt.subplots_adjust(wspace=0.25, hspace=0.25)
    plt.savefig(savePath, dpi=500)

    im = Image.open(savePath)

    width, height = im.size

    im1 = im.crop((0, 0, width, 7600))
    im1.save(savePath, dpi=(500, 500))

def plot_reg(model, workPath, savePath):
    inputTrain, outputTrain, inputTest, outputTest, inputAll, outputAll, trainingWeights = load_resampled_data()

    fig1 = plt.figure(figsize=(18, 17))
    spec1 = gridspec.GridSpec(ncols=5, nrows=6, figure=fig1)

    weightPaths = ['rNone', 'r0000001', 'r00001', 'r001', 'r1']

    dataLabels = [r'$\lambda_{\rm{low}},\dot{\lambda}_{\rm{med}}$',
                  r'$\lambda_{\rm{med}},\dot{\lambda}_{\rm{med}}$',
                  r'$\lambda_{\rm{high}},\dot{\lambda}_{\rm{med}}$',
                  r'$\lambda_{\rm{high}},\dot{\lambda}_{\rm{slow}}$',
                  r'$\lambda_{\rm{high}},\dot{\lambda}_{\rm{fast}}$']

    regLabels = [r'$\theta=0.0$', r'$\theta=1.0e$-7', r'$\theta=1.0e$-5', r'$\theta=1.0e$-3', r'$\theta=1.0e$-1']

    legendLbl = ['data', r'$[I_1-3]$', r'$exp([I_1-3])$', r'$ln(1-[I_1-3])$',
                 r'$[I_1-3]^2$', r'$exp([I_1-3]^2)$', r'$ln(1-[I_1-3]^2)$',
                 r'$[I_2-3]$', r'$exp([I_2-3])$', r'$ln(1-[I_2-3])$',
                 r'$[I_2-3]^2$', r'$exp([I_2-3]^2)$', r'$ln(1-[I_2-3]^2)$']

    for i in range(5):
        model.load_weights(workPath + weightPaths[i] + '/Train_on_All/muscleAll.tf')

        predictions = model.predict(
            [np.expand_dims(inputAll[:, :, 1], axis=2), np.expand_dims(inputAll[:, :, 0], axis=2)])

        times = []

        for j in range(len(inputAll)):
            timeArr = np.zeros((len(inputAll[j]), 1))

            count = 0
            for k in range(len(timeArr)):
                timeArr[k] = count + inputAll[j, k, 0]
                count = timeArr[k]

            times.append(timeArr)

        for j in range(5):
            ax = fig1.add_subplot(spec1[j, i])

            if j == 0:
                ax.set_title(regLabels[i], fontsize=20)

            if i == 0:
                ax.text(-120, -1.9, dataLabels[j], fontsize=20, rotation='vertical')

            mode = 'train'
            colorMode = 'c'

            inData = [np.expand_dims(inputAll[j, :, 1], axis=(0, 2)), np.expand_dims(inputAll[j, :, 0], axis=(0, 2))]
            plotFit(ax, model, inData, times[j], np.squeeze(outputAll[j]), np.squeeze(predictions[j]), colorMode,
                    savePath, mode, legendLbl)
            model.load_weights(workPath + weightPaths[i] + '/Train_on_All/muscleAll.tf')

    proxy = mpatches.Patch(color='white')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(proxy)
    labels.append(' ')
    order = [0, 13, 12, 1, 11, 2, 10, 3, 9, 4, 8, 5, 7, 6]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc=[-5.2, -.7],
               fontsize=18, ncol=int(np.ceil(len(legendLbl) / 2)))

    plt.subplots_adjust(wspace=0.25, hspace=0.25)
    plt.savefig(savePath, dpi=500)

    im = Image.open(savePath)

    width, height = im.size

    im1 = im.crop((0, 0, width, 7900))
    im1.save(savePath, dpi=(500, 500))



plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.pad'] = 14
plt.rcParams['ytick.major.pad'] = 5
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['axes.axisbelow'] = True

ColorI = [0.177423, 0.437527, 0.557565]

L2 = 0  # 0.001
rp = 0.0
numOgdenTerms = 10
numHistoryVars = 10
workPath = 'Ogden/Resample/Train_on_One/'
savePath = workPath + 'CANN_reg_figure.jpg'
saveFile = 'CANN_prony.jpg'

disable_eager_execution()

model = build_ogden(numOgdenTerms, numHistoryVars)

#model = build_CANN(numHistoryVars, L2, rp)

# plot_reg(model,workPath,savePath)

# weights = ['Ogden_muscleMag10.tf', 'Ogden_muscleMag20v2.tf', 'Ogden_muscleMag30v2.tf',
#           'Ogden_muscleRate05.tf', 'Ogden_muscleRate5v2.tf']
weights = ['Ogden_muscleMag10.tf', 'Ogden_muscleMag20.tf', 'Ogden_muscleMag30.tf',
          'Ogden_muscleRate05.tf', 'Ogden_muscleRate5.tf']

for i in range(5):
    model.load_weights(workPath+weights[i])
    print(model.get_weights())

# powers = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8, 9, -9, 10, -10]
#
# for w in weights:
#     # calculate shear modulus
#     model.load_weights(workPath + w)
#     ogParams = model.get_weights()[0]
#
#     ogSum = 0
#
#     for i in range(len(powers)):
#         ogSum = ogSum + ogParams[i]*(powers[i]**2)
#
#     mu = ogSum/2
#     print(mu)
