const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

function loadMarketFormattingHelpers() {
  const templatePath = path.join(__dirname, '..', 'templates', 'index.html');
  const html = fs.readFileSync(templatePath, 'utf8');
  const start = html.indexOf('function normalizeDisplayMarketKey');
  const end = html.indexOf('function roiBandClass', start);
  assert.notEqual(start, -1, 'normalizeDisplayMarketKey not found');
  assert.notEqual(end, -1, 'roiBandClass not found after market helpers');
  const snippet = html.slice(start, end);
  return new Function(`${snippet}\nreturn { formatLine, formatLineForMarket, isDisplayTotalMarket, isDisplaySpreadMarket };`)();
}

test('formatLineForMarket omits leading plus for totals markets', () => {
  const { formatLineForMarket, isDisplayTotalMarket } = loadMarketFormattingHelpers();

  assert.equal(isDisplayTotalMarket('totals'), true);
  assert.equal(isDisplayTotalMarket('alternate_totals'), true);
  assert.equal(isDisplayTotalMarket('totals_q1'), true);
  assert.equal(formatLineForMarket(6.5, 'totals'), '6.5');
  assert.equal(formatLineForMarket(6.5, 'alternate_totals'), '6.5');
  assert.equal(formatLineForMarket(6.5, 'totals_h1'), '6.5');
});

test('formatLineForMarket preserves signed spreads formatting', () => {
  const { formatLine, formatLineForMarket, isDisplaySpreadMarket } = loadMarketFormattingHelpers();

  assert.equal(isDisplaySpreadMarket('spreads'), true);
  assert.equal(isDisplaySpreadMarket('alternate_spreads'), true);
  assert.equal(isDisplaySpreadMarket('spreads_q2'), true);
  assert.equal(formatLine(6.5), '+6.5');
  assert.equal(formatLineForMarket(6.5, 'spreads'), '+6.5');
  assert.equal(formatLineForMarket(-1.5, 'spreads_h1'), '-1.5');
});
