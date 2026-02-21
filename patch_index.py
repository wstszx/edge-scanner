import re
import traceback

def patch_index():
    try:
        with open('templates/index.html', 'r', encoding='utf-8') as f:
            html = f.read()

        # 1. Add History nav button
        if 'data-tab="history"' not in html:
            nav_pattern = r'(<button type="button" class="tab-btn" data-tab="plus-ev" data-i18n="tab_plus_ev">\+EV</button>)'
            nav_replacement = r'\1\n      <button type="button" class="tab-btn" data-tab="history" data-i18n="tab_history">History</button>'
            html = re.sub(nav_pattern, nav_replacement, html, count=1)

        # 2. Add History panel at the end of the content section
        if 'id="history-panel"' not in html:
            panel_pattern = r'(</section>\s*</div>\s*</main>)'
            history_panel = '''
    <section id="history-panel" class="hidden">
      <section class="card results-card">
        <div class="card-head">
          <h3 data-i18n="history_log_open">History Log</h3>
          <div class="table-meta">
            <span id="history-table-count">0 records</span>
            <button type="button" id="refresh-history-btn" class="secondary-btn" data-i18n="refresh_btn">Refresh</button>
          </div>
        </div>
        <div class="table-scroll">
          <table id="history-table">
            <thead>
              <tr>
                <th data-i18n="table_time">Scan Time</th>
                <th data-i18n="table_mode">Mode</th>
                <th data-i18n="table_metric">Metric (ROI/EV/Edge)</th>
                <th data-i18n="table_sport">Sport</th>
                <th data-i18n="table_event">Event</th>
                <th data-i18n="table_market">Market</th>
                <th data-i18n="table_match">Match</th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
        <div id="history-empty" class="empty" data-i18n="empty_history">No history recorded yet. Enable history tracking and run a scan.</div>
      </section>
    </section>
'''
            html = re.sub(panel_pattern, history_panel + r'\1', html, count=1)

        # 3. Add translation strings for History
        history_i18n = '''
        tab_history: 'History',
        history_log_open: 'History Log',
        refresh_btn: 'Refresh',
        empty_history: 'No history recorded yet. Enable history tracking and run a scan.',
        table_time: 'Scan Time',
        table_mode: 'Mode',
        table_metric: 'Metric (ROI/EV/Edge)',
        table_match: 'Match',
'''
        history_i18n_zh = '''
        tab_history: '历史',
        history_log_open: '扫描历史',
        refresh_btn: '刷新',
        empty_history: '暂无历史记录。请开启历史追踪并运行扫描。',
        table_time: '扫描时间',
        table_mode: '模式',
        table_metric: '指标',
        table_match: '对阵',
'''
        if 'tab_history:' not in html:
            html = re.sub(
                r'(tab_plus_ev:\s*\'\+EV\',)', 
                r'\g<1>\n' + history_i18n, 
                html, 
                count=1
            )
            html = re.sub(
                r'(tab_plus_ev:\s*\'\+EV\',)', 
                r'\g<1>\n' + history_i18n_zh, 
                html, 
                count=1
            )

        # 4. Add History Tab logic inside a script block
        extra_js = '''
<script>
document.addEventListener('DOMContentLoaded', () => {
    const historyPanel = document.getElementById('history-panel');
    const refreshHistoryBtn = document.getElementById('refresh-history-btn');
    const historyTableBody = document.querySelector('#history-table tbody');
    const historyEmpty = document.getElementById('history-empty');
    const historyTableCount = document.getElementById('history-table-count');

    // Override the display logic for tabs to include History
    const allTabs = document.querySelectorAll('.tab-btn');
    const arbPanel = document.getElementById('arbitrage-panel');
    const arbConfig = document.getElementById('arbitrage-config');
    const midPanel = document.getElementById('middles-panel');
    const midConfig = document.getElementById('middles-config');
    const evPanel = document.getElementById('plus-ev-panel');
    const evConfig = document.getElementById('plus-ev-config');

    allTabs.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const tabId = e.target.dataset.tab;
            allTabs.forEach(t => t.classList.remove('active'));
            e.target.classList.add('active');

            arbPanel?.classList.add('hidden');
            arbConfig?.classList.add('hidden');
            midPanel?.classList.add('hidden');
            midConfig?.classList.add('hidden');
            evPanel?.classList.add('hidden');
            evConfig?.classList.add('hidden');
            historyPanel?.classList.add('hidden');

            if (tabId === 'arbitrage') {
                arbPanel?.classList.remove('hidden');
                arbConfig?.classList.remove('hidden');
            } else if (tabId === 'middles') {
                midPanel?.classList.remove('hidden');
                midConfig?.classList.remove('hidden');
            } else if (tabId === 'plus-ev') {
                evPanel?.classList.remove('hidden');
                evConfig?.classList.remove('hidden');
            } else if (tabId === 'history') {
                historyPanel?.classList.remove('hidden');
                if (window.loadHistory) window.loadHistory();
            }
        });
    });

    window.loadHistory = function loadHistory() {
      if (!refreshHistoryBtn) return;
      refreshHistoryBtn.disabled = true;
      refreshHistoryBtn.textContent = 'Loading...';
      
      fetch('/history?limit=200')
        .then(res => res.json())
        .then(data => {
          refreshHistoryBtn.disabled = false;
          refreshHistoryBtn.textContent = 'Refresh'; // Or use i18n
          
          if (!historyTableBody) return;
          historyTableBody.innerHTML = '';
          
          if (!data.success || !data.records || data.records.length === 0) {
            historyEmpty?.classList.remove('hidden');
            document.querySelector('#history-table')?.parentElement.classList.add('hidden');
            if(historyTableCount) historyTableCount.textContent = '0 records';
            return;
          }
          
          historyEmpty?.classList.add('hidden');
          document.querySelector('#history-table')?.parentElement.classList.remove('hidden');
          if(historyTableCount) historyTableCount.textContent = data.records.length + ' records';
          
          data.records.forEach(rec => {
            const tr = document.createElement('tr');
            
            let metricText = '';
            let modeTitle = rec.mode;
            let displayBooks = '';
            
            if (rec.mode === 'arbitrage') {
              metricText = `<span class="gross-roi positive-roi">${(rec.roi_percent || 0).toFixed(2)}% ROI</span>`;
              modeTitle = 'Arb';
              if (rec.books) displayBooks = rec.books.map(b => `${b.bookmaker} (${b.price})`).join(' vs ');
            } else if (rec.mode === 'ev') {
              metricText = `<span class="gross-roi positive-roi">${(rec.edge_percent || 0).toFixed(2)}% Edge</span>`;
              modeTitle = '+EV';
              displayBooks = `${rec.soft_book || ''} (${rec.soft_odds || ''}) vs Sharp (${rec.sharp_fair || ''})`;
            } else if (rec.mode === 'middles') {
              metricText = `<span class="gross-roi positive-roi">$${(rec.ev_dollars || 0).toFixed(2)} EV</span>`;
              modeTitle = 'Middle';
              if (rec.books) displayBooks = rec.books.map(b => `${b.bookmaker} (${b.price})`).join(' vs ');
            }
            
            const scanTime = rec.scan_time ? rec.scan_time.replace('T', ' ').slice(0, 19) : '';
            const sport = rec.sport_display || rec.sport || '';
            
            tr.innerHTML = `
              <td class="mono">${scanTime}</td>
              <td><span class="badge uppercase">${modeTitle}</span></td>
              <td>${metricText}</td>
              <td>${sport}</td>
              <td class="event-cell">${rec.event || ''}</td>
              <td>${rec.market || ''}</td>
              <td class="event-cell muted">${displayBooks}</td>
            `;
            historyTableBody.appendChild(tr);
          });
        })
        .catch(err => {
          console.error(err);
          if (refreshHistoryBtn) {
              refreshHistoryBtn.disabled = false;
              refreshHistoryBtn.textContent = 'Error';
          }
        });
    }

    if (refreshHistoryBtn) {
        refreshHistoryBtn.addEventListener('click', loadHistory);
    }
});
</script>
</body>
'''

        if 'History Tab Extension' not in html:
            html = html.replace('</body>', extra_js)

        with open('templates/index.html', 'w', encoding='utf-8') as f:
            f.write(html)
            
        print("Done.")

    except Exception as e:
        print("Error patching index.html:")
        traceback.print_exc()

if __name__ == '__main__':
    patch_index()
