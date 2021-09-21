import torch, os
from torch.autograd import Variable
from crnn import strLabelConverter, resizeNormalize
from PIL import Image
from filelock import FileLock
import crnn.models.crnn as crnn

model_path = os.path.join(os.getcwd(), 'crnn/data/crnn.pth')
alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def ocr(file):
    with FileLock(file + '.lock'):
        converter = strLabelConverter(alphabet)
        model = crnn.CRNN(32, 1, 37, 256)

        model.load_state_dict(torch.load(model_path))

        transformer = resizeNormalize((100, 32))
        image = Image.open(file).convert('L')
        image = transformer(image)
        if torch.cuda.is_available():
            image = image.cuda()
        image = image.view(1, *image.size())
        image = Variable(image)

        model.eval()
        preds = model(image)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        # raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
        return sim_pred