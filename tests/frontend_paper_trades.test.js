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
    "  paper_trade_invalid: 'Invalid',",
    "  paper_trade_settled: 'Settled',",
    "  paper_trade_recorded: 'Recorded',",
    "  paper_trade_no_ticket: 'No ticket',",
    "  paper_trade_settlement_pnl: 'Settled PnL {amount}',",
    "  paper_trade_settlement_leg: '{result} · PnL {amount}',",
    "  paper_trade_blockers: 'Blockers: {items}',",
    "  paper_trade_none: 'No paper trades recorded yet.',",
    "  paper_trade_type_arbitrage: 'Arbitrage',",
    "  paper_trade_type_middle: 'Middle',",
    "  paper_trade_type_plus_ev: '+EV',",
    "  paper_trade_scan_summary: 'Scan paper: {created} new, {ready} ready, {blocked} blocked · Arb {arbReady}/{arbBlocked} · Middle {middleReady}/{middleBlocked}',",
    "  paper_trade_leg_outcome: 'Buy {outcome}',",
    "  paper_trade_leg_line: 'Line {line}',",
    "  paper_trade_leg_price: 'Limit {price}',",
    "  paper_trade_leg_stake: 'Stake {amount}',",
    "  paper_trade_leg_max_stake: 'Max {amount}',",
    "  paper_trade_leg_fee: 'Fee {rate}',",
    "  paper_trade_leg_quote: 'Quote {time}',",
    "  paper_trade_leg_id: 'ID {id}',",
    "  paper_trade_leg_manual_web: 'Manual web',",
    "  paper_trade_leg_open: 'Open book',",
    "  paper_trade_leg_risk_artline_api_max_bet_below_min_bet: 'API max bet below minimum',",
    "  paper_trade_leg_risk_manual_liquidity_unverified: 'Manual liquidity unverified',",
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
    "return { paperTradeStatusMeta, renderPaperTradeStatusCell, renderPaperTradeListHtml, renderPaperTradeSummaryHtml };",
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
  assert.deepEqual(
    paperTradeStatusMeta({ status: 'invalid', submit_blockers: ['same_team_spread_middle'] }),
    { label: 'Invalid', tone: 'invalid', title: 'Blockers: same_team_spread_middle' }
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

test('paper trade list renders recorded plus EV legs', () => {
  const { renderPaperTradeListHtml } = loadPaperTradeHelpers();
  const html = renderPaperTradeListHtml([
    {
      created_at: '2026-05-26T08:40:00Z',
      execution_type: 'plus_ev',
      event: 'Los Angeles Lakers vs Boston Celtics',
      market: 'h2h',
      edge_percent: 8.5,
      ev_per_100: 8.5,
      total_stake: 42.5,
      ticket: {
        execution_type: 'plus_ev',
        status: 'paper_ready',
        paper_trade_ready: true,
        legs: [
          { outcome: 'Los Angeles Lakers', bookmaker_key: 'polymarket', bookmaker: 'Polymarket', stake: 42.5, limit_price: 2.25 },
        ],
      },
    },
  ]);

  assert.match(html, /\+EV/);
  assert.match(html, /Los Angeles Lakers vs Boston Celtics/);
  assert.match(html, /Polymarket/);
  assert.match(html, /8\.50%/);
});

test('paper trade list explains each execution leg with identifiers and limits', () => {
  const { renderPaperTradeListHtml } = loadPaperTradeHelpers();
  const html = renderPaperTradeListHtml([
    {
      created_at: '2026-05-26T08:40:00Z',
      execution_type: 'arbitrage',
      event: 'Los Angeles Lakers vs Boston Celtics',
      market: 'spreads',
      roi_percent: 1.25,
      total_stake: 100,
      ticket: {
        execution_type: 'arbitrage',
        status: 'paper_ready',
        paper_trade_ready: true,
        legs: [
          {
            outcome: 'Los Angeles Lakers',
            bookmaker_key: 'sx_bet',
            bookmaker: 'SX Bet',
            point: 3.5,
            stake: 48.78,
            max_stake: 784.62,
            limit_price: 1.995,
            fee_rate: 0,
            market_hash: '0xsxbet',
            quote_updated_at: '2026-05-27T06:03:54Z',
          },
          {
            outcome: 'Boston Celtics',
            bookmaker_key: 'polymarket',
            bookmaker: 'Polymarket',
            point: -3.5,
            stake: 51.22,
            max_stake: 1588,
            limit_price: 2,
            fee_rate: 0.03,
            token_id: 'poly-token',
            quote_updated_at: '2026-05-27T06:03:55Z',
          },
        ],
      },
    },
  ]);

  assert.match(html, /Buy Los Angeles Lakers/);
  assert.match(html, /Line \+3\.5/);
  assert.match(html, /Limit 1\.995/);
  assert.match(html, /Stake \$48\.78/);
  assert.match(html, /Max \$784\.62/);
  assert.match(html, /Fee 0\.00%/);
  assert.match(html, /ID 0xsxbet/);
  assert.match(html, /Buy Boston Celtics/);
  assert.match(html, /Fee 3\.00%/);
  assert.match(html, /ID poly-token/);
});

test('paper trade list highlights manual Artline web execution legs', () => {
  const { renderPaperTradeListHtml } = loadPaperTradeHelpers();
  const html = renderPaperTradeListHtml([
    {
      created_at: '2026-05-27T09:40:00Z',
      execution_type: 'arbitrage',
      event: 'Filippo Callerio vs Edoardo Zanada',
      market: 'h2h',
      roi_percent: 0.56,
      total_stake: 25,
      ticket: {
        execution_type: 'arbitrage',
        status: 'paper_ready',
        paper_trade_ready: true,
        legs: [
          {
            outcome: 'Edoardo Zanada',
            bookmaker_key: 'polymarket',
            bookmaker: 'Polymarket',
            stake: 21.25,
            limit_price: 1.186444,
            token_id: 'poly-token',
          },
          {
            outcome: 'Filippo Callerio',
            bookmaker_key: 'artline',
            bookmaker: 'Artline',
            stake: 3.75,
            limit_price: 6.6,
            book_event_url: 'https://artline.bet/bookmaker/match/prematch/tennis/385781862425200',
            draft_order: {
              adapter: 'manual_artline',
              order_type: 'manual_web',
              event_url: 'https://artline.bet/bookmaker/match/prematch/tennis/385781862425200',
            },
            manual_liquidity_risks: [
              'artline_api_max_bet_below_min_bet',
              'manual_liquidity_unverified',
            ],
          },
        ],
      },
    },
  ]);

  assert.match(html, /Manual web/);
  assert.match(html, /Open book/);
  assert.match(html, /https:\/\/artline\.bet\/bookmaker\/match\/prematch\/tennis\/385781862425200/);
  assert.match(html, /API max bet below minimum/);
  assert.match(html, /Manual liquidity unverified/);
  assert.match(html, /Buy Filippo Callerio/);
});

test('paper trade list renders settlement pnl and leg outcomes', () => {
  const { renderPaperTradeListHtml } = loadPaperTradeHelpers();
  const html = renderPaperTradeListHtml([
    {
      created_at: '2026-05-26T08:40:00Z',
      execution_type: 'arbitrage',
      event: 'Los Angeles Lakers vs Boston Celtics',
      market: 'h2h',
      roi_percent: 7.36,
      total_stake: 100,
      status: 'settled',
      settlement: {
        status: 'settled',
        pnl: 5,
        legs: [
          { outcome: 'Los Angeles Lakers', result: 'win', pnl: 55 },
          { outcome: 'Boston Celtics', result: 'loss', pnl: -50 },
        ],
      },
      ticket: {
        execution_type: 'arbitrage',
        status: 'paper_ready',
        paper_trade_ready: true,
        legs: [
          { outcome: 'Los Angeles Lakers', bookmaker_key: 'sx_bet', bookmaker: 'SX Bet', stake: 50, limit_price: 2.1 },
          { outcome: 'Boston Celtics', bookmaker_key: 'polymarket', bookmaker: 'Polymarket', stake: 50, limit_price: 2.0 },
        ],
      },
    },
  ]);

  assert.match(html, /Settled/);
  assert.match(html, /Settled PnL \$5\.00/);
  assert.match(html, /win · PnL \$55\.00/);
  assert.match(html, /loss · PnL \$-50\.00/);
});

test('paper trade summary renders current scan counts', () => {
  const { renderPaperTradeSummaryHtml } = loadPaperTradeHelpers();
  const html = renderPaperTradeSummaryHtml({
    created_count: 1,
    ready_count: 2,
    blocked_count: 3,
    arbitrage_ready_count: 0,
    arbitrage_blocked_count: 4,
    middle_ready_count: 2,
    middle_blocked_count: 1,
  });

  assert.match(html, /1 new/);
  assert.match(html, /2 ready/);
  assert.match(html, /3 blocked/);
  assert.match(html, /Arb 0\/4/);
  assert.match(html, /Middle 2\/1/);
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
