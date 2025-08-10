// preprocess.js
function fixZeros(row, medianCols, medians) {
    const copy = [...row];
    medianCols.forEach(colIdx => {
        if (copy[colIdx] === 0) {
            copy[colIdx] = medians[colIdx];
        }
    });
    return copy;
}

function minMaxNormalizeSingle(row, mins, maxs) {
    return row.map((v, i) => (v - mins[i]) / (maxs[i] - mins[i]));
}

module.exports = { fixZeros, minMaxNormalizeSingle };
