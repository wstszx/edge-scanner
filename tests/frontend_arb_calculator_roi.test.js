const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

function loadArbCalculatorSummaryHelpers() {
  const templatePath = path.join(__dirname, '..', 'templates', 'index.html');
  const html = fs.readFileSync(templatePath, 'utf8');
  const start = html.indexOf('function roundMoney');
  const end = html.indexOf('function parseStakeValue', start);
  assert.notEqual(start, -1, 'roundMoney not found');
  assert.notEqual(end, -1, 'parseStakeValue not found after roundMoney');
  const snippet = html.slice(start, end);
  return new Function(snippet + '\nreturn { roundMoney, calculateArbCalculatorSummary };')();
}

test('arb calculator ROI uses unrounded profit before display rounding', () => {
  const { calculateArbCalculatorSummary } = loadArbCalculatorSummaryHelpers();

  const summary = calculateArbCalculatorSummary(
    [
      { base: 1.58, baseExact: 1.584 },
      { base: 1.60, baseExact: 1.5972 },
    ],
    1.57
  );

  assert.equal(summary.minPayout.toFixed(2), '1.58');
  assert.equal(summary.profit.toFixed(2), '0.01');
  assert.equal(summary.roi.toFixed(2), '0.89');
});