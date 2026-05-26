const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

function loadPaperTradeHelpers() {
  const templatePath = path.join(__dirname, '..', 'templates', 'index.html');
  const html = fs.readFileSync(templatePath, 'utf8');
  const start = html.indexOf('function paperTradeStatusMeta');
  const end = html.indexOf('function renderPlusEv()', start);
  assert.notEqual(start, -1, 'paperTradeStatusMeta not found');
  assert.notEqual(end, -1, 'renderPlusEv not found after paper trade helpers');
  const snippet = html.slice(start, end);
  const source = [
    "const translations = {",
    "  paper_trade_ready: 'Paper ready',",
    "  paper_trade_blocked: 'Blocked',",
    "  paper_trade_recorded: 'Recorded',",
    "  paper_trade_no_ticket: 'No ticket',",
    "  paper_trade_blockers: 'Blockers: {items}',",
    "  paper_trade_none: 'No paper trades recorded yet.',",
    "  paper_trade_type_arbitrage: 'Arbitrage',",
    "  paper_trade_type_middle: 'Middle',",
    "};",
    "const t = (key, params = {}) => {",
    "  let text = translations[key] || key;",
    "  for (const [param, value] of Object.entries(params)) text = text.replace('{' + param + '}', String(value));",
    "  return text;",
    "};",
    "const escapeHtml = (value) => String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\"/g, '&quot;').replace(/'/g, '&#39;');",
    "const formatPercent = (value) => `${Number(value || 0).toFixed(2)}%`;",
    "const formatCurrency = (value) => `$${Number(value || 0).toFixed(2)}`;",
    "const formatDateTime = (value) => String(value || '');",
    "const formatLineForMarket = (value) => Number(value).toFixed(1);",
    "const formatOddsDisplay = (value) => Number(value).toFixed(3);",
    snippet,
    "return { paperTradeStatusMeta, renderPaperTradeStatusCell, renderPaperTradeListHtml };",
  ].join('\n');
  return new Function(source)();
}

function loadArbPaperTicketHelpers() {
  const templatePath = path.join(__dirname, '..', 'templates', 'index.html');
  const html = fs.readFileSync(templatePath, 'utf8');
  const start = html.indexOf('function arbTicketSignature');
  const end = html.indexOf('function normalizeArbLegToken', start);
  assert.notEqual(start, -1, 'arbTicketSignature not found');
  assert.notEqual(end, -1, 'normalizeArbLegToken not found after arb helpers');
  const snippet = html.slice(start, end);
  const source = [
    "let lastScanData = { execution_tickets: [",
    "  { opportunity_id: 'event-1', event: 'Los Angeles Lakers vs Boston Celtics', market: 'h2h', legs: [] },",
    "] };",
    snippet,
    "return { arbTicketSignature, arbTicketMatchesOpportunity, findArbExecutionTicket };",
  ].join('\n');
  return new Function(source)();
}

test('paper trade status labels ready and blocked tickets', () => {
  const { paperTradeStatusMeta } = loadPaperTradeHelpers();

  assert.deepEqual(
    paperTradeStatusMeta({ status: 'paper_ready', paper_trade_ready: true }),
    { label: 'Paper ready', tone: 'ready', title: 'Paper ready' }
  );
  assert.deepEqual(
    paperTradeStatusMeta({ status: 'blocked', submit_blockers: ['quote_time_skew'] }),
    { label: 'Blocked', tone: 'blocked', title: 'Blockers: quote_time_skew' }
  );
});

test('paper trade list renders recorded middle legs', () => {
  const { renderPaperTradeListHtml } = loadPaperTradeHelpers();
  const html = renderPaperTradeListHtml([
    {
      created_at: '2026-05-26T08:40:00Z',
      event: 'San Antonio Spurs vs Oklahoma City Thunder',
      market: 'spreads',
      middle_zone: 'Spurs by 4-5',
      ev_percent: 23.07,
      total_stake: 100,
      ticket: {
        legs: [
          { bookmaker_key: 'polymarket', bookmaker: 'Polymarket', line: -3.5, stake: 37.59, limit_price: 3.125 },
          { bookmaker_key: 'sx_bet', bookmaker: 'SX Bet', line: 5.5, stake: 62.41, limit_price: 1.877934 },
        ],
      },
    },
  ]);

  assert.match(html, /San Antonio Spurs vs Oklahoma City Thunder/);
  assert.match(html, /Polymarket/);
  assert.match(html, /SX Bet/);
  assert.match(html, /23\.07%/);
});

test('paper trade list renders recorded arbitrage legs', () => {
  const { renderPaperTradeListHtml } = loadPaperTradeHelpers();
  const html = renderPaperTradeListHtml([
    {
      created_at: '2026-05-26T08:40:00Z',
      execution_type: 'arbitrage',
      event: 'Los Angeles Lakers vs Boston Celtics',
      market: 'h2h',
      roi_percent: 7.36,
      total_stake: 100,
      ticket: {
        execution_type: 'arbitrage',
        status: 'paper_ready',
        paper_trade_ready: true,
        legs: [
          { outcome: 'Los Angeles Lakers', bookmaker_key: 'sx_bet', bookmaker: 'SX Bet', stake: 48.78, limit_price: 2.2 },
          { outcome: 'Boston Celtics', bookmaker_key: 'polymarket', bookmaker: 'Polymarket', stake: 51.22, limit_price: 2.1 },
        ],
      },
    },
  ]);

  assert.match(html, /Arbitrage/);
  assert.match(html, /Los Angeles Lakers vs Boston Celtics/);
  assert.match(html, /SX Bet/);
  assert.match(html, /Polymarket/);
  assert.match(html, /7\.36%/);
});

test('arbitrage paper ticket matcher uses event ids and leg signature fallback', () => {
  const { arbTicketMatchesOpportunity, findArbExecutionTicket } = loadArbPaperTicketHelpers();
  assert.equal(findArbExecutionTicket({ id: 'other', event_id: 'event-1' }).opportunity_id, 'event-1');
  assert.equal(
    arbTicketMatchesOpportunity(
      {
        event: 'Los Angeles Lakers vs Boston Celtics',
        market: 'h2h',
        legs: [
          { bookmaker_key: 'sx_bet', outcome: 'Los Angeles Lakers' },
          { bookmaker_key: 'polymarket', outcome: 'Boston Celtics' },
        ],
      },
      {
        event: 'Los Angeles Lakers vs Boston Celtics',
        market: 'h2h',
        best_odds: [
          { bookmaker_key: 'sx_bet', outcome: 'Los Angeles Lakers' },
          { bookmaker_key: 'polymarket', outcome: 'Boston Celtics' },
        ],
      }
    ),
    true
  );
});
