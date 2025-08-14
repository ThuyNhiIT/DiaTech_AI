// train.js
const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');

(async () => {
    console.log("Training model (robust pipeline)...");

    // ===== 0) Utils với seed để lặp lại =====
    const SEED = 2025;
    function mulberry32(a) {
        return function () {
            let t = a += 0x6D2B79F5;
            t = Math.imul(t ^ t >>> 15, t | 1);
            t ^= t + Math.imul(t ^ t >>> 7, t | 61);
            return ((t ^ t >>> 14) >>> 0) / 4294967296;
        };
    }
    const rand = mulberry32(SEED);
    function seededShuffle(arr) {
        for (let i = arr.length - 1; i > 0; i--) {
            const j = Math.floor(rand() * (i + 1));
            [arr[i], arr[j]] = [arr[j], arr[i]];
        }
    }

    // ===== 1) Đọc CSV =====
    const csvPath = path.join(__dirname, 'data', 'pima_diabetes.csv');
    const csvDataset = tf.data.csv(`file://${csvPath}`, { hasHeader: true });
    const rows = await csvDataset.toArray();
    let data = rows.map(r => Object.values(r).map(Number));
    if (!data.length) {
        throw new Error('CSV rỗng hoặc không đọc được.');
    }

    // ===== 2) X, y =====
    const Xall = data.map(d => d.slice(0, 8));
    const yall = data.map(d => d[8]);

    // ===== 3) Stratified split (train/val/test = 70/10/20) =====
    const idx0 = [], idx1 = [];
    yall.forEach((y, i) => (y === 1 ? idx1 : idx0).push(i));
    // shuffle theo seed
    seededShuffle(idx0); seededShuffle(idx1);

    function takeSplit(idxs, fracTrain, fracVal) {
        const n = idxs.length;
        const nTrain = Math.floor(n * fracTrain);
        const nVal = Math.floor(n * fracVal);
        return {
            train: idxs.slice(0, nTrain),
            val: idxs.slice(nTrain, nTrain + nVal),
            test: idxs.slice(nTrain + nVal)
        };
    }
    const fracTrain = 0.7, fracVal = 0.1;
    const s0 = takeSplit(idx0, fracTrain, fracVal);
    const s1 = takeSplit(idx1, fracTrain, fracVal);

    const trainIdx = s0.train.concat(s1.train);
    const valIdx = s0.val.concat(s1.val);
    const testIdx = s0.test.concat(s1.test);

    // shuffle gộp
    seededShuffle(trainIdx);
    seededShuffle(valIdx);
    seededShuffle(testIdx);

    function gather(idxArr) {
        return {
            X: idxArr.map(i => Xall[i].slice()),
            y: idxArr.map(i => yall[i])
        };
    }
    let { X: Xtrain, y: ytrain } = gather(trainIdx);
    let { X: Xval, y: yval } = gather(valIdx);
    let { X: Xtest, y: ytest } = gather(testIdx);

    // ===== 4) Tiền xử lý (TRAIN) =====
    // 4.1 Sửa 0 bằng median tại các cột có 0 giả
    const medianCols = [1, 2, 3, 4, 5]; // Glucose, BP, Skin, Insulin, BMI
    function computeMedians(arr, cols) {
        const meds = {};
        cols.forEach(ci => {
            const v = arr.map(r => r[ci]).filter(x => x !== 0);
            v.sort((a, b) => a - b);
            const m = v.length ? (v.length % 2 ? v[(v.length - 1) / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2) : 0;
            meds[ci] = m || 0;
        });
        return meds;
    }
    const medians = computeMedians(Xtrain, medianCols);
    function fixZerosInPlace(arr) {
        arr.forEach(row => {
            medianCols.forEach(ci => {
                if (row[ci] === 0) row[ci] = medians[ci];
            });
        });
    }
    fixZerosInPlace(Xtrain);
    fixZerosInPlace(Xval);
    fixZerosInPlace(Xtest);

    // 4.2 Winsorization theo IQR (tính trên TRAIN)
    function iqrBounds(arr) {
        const lower = [], upper = [];
        const D = arr[0].length;
        for (let j = 0; j < D; j++) {
            const col = arr.map(r => r[j]).slice().sort((a, b) => a - b);
            const q1 = quantile(col, 0.25);
            const q3 = quantile(col, 0.75);
            const iqr = q3 - q1;
            lower[j] = q1 - 1.5 * iqr;
            upper[j] = q3 + 1.5 * iqr;
        }
        return { lower, upper };
    }
    function quantile(sortedArr, q) {
        const pos = (sortedArr.length - 1) * q;
        const base = Math.floor(pos), rest = pos - base;
        if (sortedArr[base + 1] !== undefined) {
            return sortedArr[base] + rest * (sortedArr[base + 1] - sortedArr[base]);
        } else {
            return sortedArr[base];
        }
    }
    const { lower: iqrLower, upper: iqrUpper } = iqrBounds(Xtrain);
    function winsorizeInPlace(arr) {
        arr.forEach(row => {
            for (let j = 0; j < row.length; j++) {
                if (row[j] < iqrLower[j]) row[j] = iqrLower[j];
                if (row[j] > iqrUpper[j]) row[j] = iqrUpper[j];
            }
        });
    }
    winsorizeInPlace(Xtrain);
    winsorizeInPlace(Xval);
    winsorizeInPlace(Xtest);

    // 4.3 Min–Max Normalize (fit trên TRAIN)
    function fitMinMax(arr) {
        const D = arr[0].length;
        const mins = new Array(D).fill(Infinity);
        const maxs = new Array(D).fill(-Infinity);
        arr.forEach(r => {
            for (let j = 0; j < D; j++) {
                if (r[j] < mins[j]) mins[j] = r[j];
                if (r[j] > maxs[j]) maxs[j] = r[j];
            }
        });
        return { mins, maxs };
    }
    function transformMinMax(arr, mins, maxs) {
        return arr.map(r => r.map((v, j) => maxs[j] === mins[j] ? 0 : (v - mins[j]) / (maxs[j] - mins[j])));
    }
    const { mins, maxs } = fitMinMax(Xtrain);
    Xtrain = transformMinMax(Xtrain, mins, maxs);
    Xval = transformMinMax(Xval, mins, maxs);
    Xtest = transformMinMax(Xtest, mins, maxs);

    // ===== 5) Tensor hóa =====
    const XtrainT = tf.tensor2d(Xtrain);
    const ytrainT = tf.tensor2d(ytrain, [ytrain.length, 1]);
    const XvalT = tf.tensor2d(Xval);
    const yvalT = tf.tensor2d(yval, [yval.length, 1]);
    const XtestT = tf.tensor2d(Xtest);
    const ytestT = tf.tensor2d(ytest, [ytest.length, 1]);

    // ===== 6) Model (đơn giản, regularize, early-stopping) =====
    const model = tf.sequential();
    model.add(tf.layers.dense({
        inputShape: [8],
        units: 32,
        activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({ l2: 1e-4 })
    }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({
        units: 16,
        activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({ l2: 1e-4 })
    }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    model.compile({
        optimizer: tf.train.adam(1e-3),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy'],
    });

    let bestValLoss = Infinity;
    let bestWeights = null;
    let patience = 20;
    let wait = 0;
    const EPOCHS = 300;
    const BATCH = 32;

    await model.fit(XtrainT, ytrainT, {
        epochs: EPOCHS,
        batchSize: BATCH,
        validationData: [XvalT, yvalT],
        shuffle: true,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                const acc = logs.acc ?? logs.accuracy;
                const valAcc = logs.val_acc ?? logs.val_accuracy;
                console.log(`Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)}, acc=${acc?.toFixed(4)}, val_loss=${logs.val_loss.toFixed(4)}, val_acc=${valAcc?.toFixed(4)}`);
                if (logs.val_loss + 1e-6 < bestValLoss) {
                    bestValLoss = logs.val_loss;
                    wait = 0;
                    bestWeights = model.getWeights().map(w => w.clone());
                } else {
                    wait += 1;
                    if (wait >= patience) {
                        console.log(`Early stopping at epoch ${epoch + 1}. Best val_loss=${bestValLoss.toFixed(4)}`);
                        model.setWeights(bestWeights);
                        this.stopTraining = true;
                        // tfjs không có stopTraining sẵn cho callback user; thay bằng return & bỏ epochs còn lại
                    }
                }
            }
        }
    });

    // ===== 7) Chọn ngưỡng tối ưu theo Youden trên VAL =====
    const valPred = model.predict(XvalT);
    const valProb = Array.from(await valPred.data());
    const thresholds = Array.from({ length: 101 }, (_, i) => i / 100);
    function metricsAtThreshold(prob, yTrue, th) {
        let tp = 0, fp = 0, tn = 0, fn = 0;
        for (let i = 0; i < prob.length; i++) {
            const p = prob[i] >= th ? 1 : 0;
            if (p === 1 && yTrue[i] === 1) tp++;
            else if (p === 1 && yTrue[i] === 0) fp++;
            else if (p === 0 && yTrue[i] === 0) tn++;
            else fn++;
        }
        const tpr = tp / (tp + fn || 1);
        const fpr = fp / (fp + tn || 1);
        const prec = tp / (tp + fp || 1);
        const rec = tpr;
        const f1 = (2 * prec * rec) / (prec + rec || 1);
        return { tp, fp, tn, fn, tpr, fpr, prec, rec, f1 };
    }
    let bestTh = 0.5, bestJ = -Infinity;
    for (const th of thresholds) {
        const m = metricsAtThreshold(valProb, yval, th);
        const J = m.tpr - m.fpr; // Youden's J
        if (J > bestJ) {
            bestJ = J; bestTh = th;
        }
    }
    console.log(`Best threshold (Youden) on VAL = ${bestTh.toFixed(2)} (J=${bestJ.toFixed(3)})`);

    // ===== 8) Đánh giá trên TEST =====
    const testPred = model.predict(XtestT);
    const testProb = Array.from(await testPred.data());

    // ROC-AUC
    function rocAuc(prob, yTrue) {
        // sort by prob desc
        const pairs = prob.map((p, i) => ({ p, y: yTrue[i] })).sort((a, b) => b.p - a.p);
        let P = yTrue.reduce((s, v) => s + (v === 1), 0);
        let N = yTrue.length - P;
        let tp = 0, fp = 0;
        // Construct points and integrate (trapezoid)
        let prevFpr = 0, prevTpr = 0, auc = 0;
        let lastP = Infinity;
        for (let i = 0; i < pairs.length; i++) {
            if (pairs[i].p !== lastP) {
                const tpr = tp / (P || 1);
                const fpr = fp / (N || 1);
                auc += (fpr - prevFpr) * (tpr + prevTpr) / 2;
                prevFpr = fpr; prevTpr = tpr;
                lastP = pairs[i].p;
            }
            if (pairs[i].y === 1) tp++; else fp++;
        }
        // last step to (1,1)
        const tpr = tp / (P || 1);
        const fpr = fp / (N || 1);
        auc += (fpr - prevFpr) * (tpr + prevTpr) / 2;
        return auc;
    }
    const auc = rocAuc(testProb, ytest);

    const M = metricsAtThreshold(testProb, ytest, bestTh);
    const acc = (M.tp + M.tn) / (M.tp + M.tn + M.fp + M.fn);
    console.log(`\n===== TEST METRICS @ th=${bestTh.toFixed(2)} =====`);
    console.log(`Accuracy : ${(acc * 100).toFixed(2)}%`);
    console.log(`Precision: ${(M.prec * 100).toFixed(2)}%`);
    console.log(`Recall   : ${(M.rec * 100).toFixed(2)}%`);
    console.log(`F1-score : ${(M.f1 * 100).toFixed(2)}%`);
    console.log(`ROC-AUC  : ${auc.toFixed(4)}\n`);

    // ===== 9) Lưu model + tham số tiền xử lý =====
    const savePath = path.join(__dirname, 'model');
    if (!fs.existsSync(savePath)) fs.mkdirSync(savePath);

    await model.save(tf.io.withSaveHandler(async (artifacts) => {
        fs.writeFileSync(path.join(savePath, 'model.json'), JSON.stringify(artifacts));
        if (artifacts.weightData) {
            fs.writeFileSync(path.join(savePath, 'weights.bin'), Buffer.from(artifacts.weightData));
        }
        return { modelArtifactsInfo: { dateSaved: new Date() } };
    }));

    const params = {
        seed: SEED,
        medianCols,
        medians,
        iqrLower,
        iqrUpper,
        mins,
        maxs,
        threshold: bestTh
    };
    fs.writeFileSync(path.join(savePath, 'minmax.json'), JSON.stringify(params, null, 2));

    console.log('Model + preprocessing params saved to /model.');
})();
