"""Rebuild the standalone window notebook from the pinned TFT snapshot + cells.

Run: python kaggle/build_m5_window_notebook.py
No training or data loading is performed by this builder.
"""
import ast
import hashlib
import json
from pathlib import Path


def build():
    root = Path(__file__).resolve().parent
    base_path = root / 'tft_m5_zero_test_improved.ipynb'
    template_path = root / 'm5_window_experiment_cells.py'
    base_text = base_path.read_text(encoding='utf-8')
    template = template_path.read_text(encoding='utf-8')
    original = json.loads(base_text)

    def source(index):
        return ''.join(original['cells'][index]['source'])

    def definitions(index):
        return '\n\n'.join(ast.unparse(node) for node in ast.parse(source(index)).body
                           if isinstance(node, (ast.FunctionDef, ast.ClassDef)))

    sections = {}
    for section in template.split('# %% ')[1:]:
        name, code = section.split('\n', 1)
        sections[name.strip()] = code
    fingerprint = hashlib.sha256((base_text + template + Path(__file__).read_text(encoding='utf-8')).encode()).hexdigest()
    cells = []

    def markdown(text):
        cells.append({'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)})

    def code(text):
        ast.parse(text)
        cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
                      'source':text.splitlines(keepends=True)})

    markdown('''# M5 TFT — Kiểm chứng độ dài cửa sổ encoder

Notebook độc lập, nhúng bản chụp feature/model/loss từ `tft_m5_zero_test_improved.ipynb`.
Không cần notebook gốc hoặc các file Python đi kèm khi chạy trên Kaggle.

**Câu hỏi:** với bộ feature hiện tại, encoder 14/28/56/84/112 ngày có giúp dự báo tốt hơn 56 ngày không?
Horizon cố định 28 ngày; cùng origins, batch và feature bank. Chỉ đọc sales đến d_1913.

## Cách chạy
- Dependencies: numpy, pandas, torch, matplotlib. Kaggle GPU được khuyến nghị cho full M5.
- Đặt `M5_DATA_ROOT` nếu cần; tự tìm Kaggle input và data/raw.
- Mặc định full: 30.490 series, 3 seeds, tối đa 20 epoch, 160 origins, batch 256.
- Kiểm tra nhanh: đặt `WINDOW_SMOKE=1`; chỉ dùng để kiểm tra pipeline.
- Ví dụ trong cell riêng trước config: `%env WINDOW_SMOKE=1` và `%env WINDOW_LENGTHS=28,56`.
- Chạy seed đầu trước: `%env WINDOW_RUN_SEEDS=42`, giữ WINDOW_SEEDS mặc định để khóa grid 3 seed.
- Sau đó xóa WINDOW_RUN_SEEDS hoặc đặt `42,123,2026`: các seed/lượt hoàn tất được resume, chỉ chạy lượt thiếu.
- Thay đổi grid WINDOW_SEEDS hoặc WINDOW_LENGTHS cần OUT_DIR mới do manifest khóa cấu hình.
- Output: `WINDOW_OUT_DIR` (mặc định `window_experiment_outputs`).
- `WINDOW_CONFIRM_CUTOFFS=1801,1829` bật các origin kiểm tra độ nhạy; mỗi cutoff có inner validation riêng.

**Protocol:** inner train đến d_1857 → chọn epoch trên d_1858–1885 → khởi tạo lại/refit đến d_1885 → chấm outer validation d_1886–1913.
Metric chính: WRMSSE-5 BU chưa calibration. Không dùng outer validation để early stopping.
WRMSSE-5 không phải WRMSSE 12 cấp official M5.
''')
    code(sections['config'])
    markdown('## 1. Helpers nhúng độc lập\nCác module evaluator và zero-aware features được tạo trong bộ nhớ, không ghi file helper ra thư mục chạy.')
    helper_code = 'import types\n'
    for index, name in [(3, '_window_wrmsse5'), (4, '_window_hurdle_features')]:
        helper_source = source(index).split('\n', 1)[1]
        helper_code += f"module = types.ModuleType({name!r})\nexec({helper_source!r}, module.__dict__)\nsys.modules[{name!r}] = module\n"
    helper_code += 'from _window_wrmsse5 import LEVELS, build_contexts, evaluate_wrmsse5\n'
    helper_code += 'from _window_hurdle_features import HIER_CHANNELS, LOCAL_STATE_COLS, build_hierarchy_context, build_local_state_features, fit_feature_scaler, hierarchy_feature_names\n'
    helper_code += f'SOURCE_HASH = {fingerprint!r}\n'
    code(helper_code)
    markdown('## 2. Sales, calendar, giá và feature bank\nGiữ nguyên cách tạo feature TFT improved. Chỉ nạp d_1–1913; không nạp test d_1914–1941. Selection seed cố định 42. Subset aggregates không phải tổng M5 đầy đủ.')
    load = source(9).replace(
        'pd.read_csv(DATA_ROOT / SALES_FILE, dtype={"id": str})',
        'pd.read_csv(DATA_ROOT / SALES_FILE, dtype={"id": str}, usecols=lambda c: not c.startswith("d_") or int(c[2:]) <= OUTER_END)')
    code(load)
    code(source(11))
    code(source(13))
    markdown('## 3. Model và loss\nCùng TFTLite occurrence/severity heads, Tweedie + zero loss + hierarchical auxiliary loss. Không calibration. Bỏ phần tạo model/trainer tự động của notebook gốc.')
    code(definitions(17))
    training_definitions = definitions(19).replace(
        'scaler.step(optimizer)\n                scaler.update()\n                if scheduler is not None:',
        'old_scale = scaler.get_scale()\n                scaler.step(optimizer)\n                scaler.update()\n                if scheduler is not None and scaler.get_scale() >= old_scale:')
    assert 'old_scale = scaler.get_scale()' in training_definitions
    code(training_definitions)
    markdown('''## 4. Origins chung, scaler theo cutoff và manifest
Encoder ngắn dùng đúng phần suffix của encoder dài; target và future features giống hệt nhau.
Lag_364 và rolling vẫn được giữ: SEQ không phải giới hạn toàn bộ lịch sử thô mà feature từng sử dụng.
Scaler tính theo train từng cutoff và chuẩn hóa khi lấy mẫu, tránh nhân đôi panel cho mỗi window.

Manifest khóa code/data/config/environment. Giữ WINDOWS cố định khi resume; chạy lại sẽ bỏ qua các stage hoàn tất.
Nếu đổi seed grid/window/batch/cutoff, dùng OUT_DIR mới. Không tự thay batch khi OOM vì hierarchical loss phụ thuộc batch.
''')
    code(sections['experiment'])
    markdown('## 5. Preflight\nKiểm tra target chung, suffix encoder, shape và một bước backward thật trên cửa sổ dài nhất. Chọn batch chung trước khi chạy thí nghiệm. OOM: giảm WINDOW_BATCH rồi dùng OUT_DIR mới.')
    code(sections['preflight'])
    markdown('''## 6. Huấn luyện và resume
Mỗi seed/window có stage chọn epoch và refit độc lập. Checkpoint không được chọn trên outer validation.
LR OneCycle dùng cùng ngân sách MAX_EPOCHS; refit chỉ chạy prefix số epoch đã chọn.
Không lưu model của tất cả lượt để tiết kiệm disk; lưu predictions và metric. Stage đang dở sẽ chạy lại từ đầu.
Mỗi lượt hoàn tất ghi status cuối cùng; stage lỗi được ghi rõ và dừng để không âm thầm thiếu cấu hình.
''')
    code(sections['run'])
    markdown('''## 7. Đóng góp, chi phí và kết luận
Improvement% = 100 × (WRMSSE56 − WRMSSEseq) / WRMSSE56; dương là tốt hơn 56 ngày.
So sánh paired seeds, từng cấp, zero buckets và tuần dự báo. Error bars là standard deviation giữa seed, không phải CI.
Không mặc nhiên lấy điểm thấp nhất làm bằng chứng ưu thế: cần xét độ ổn định và chi phí. Khi chênh lệch nhỏ/không ổn định, cân nhắc cửa sổ ngắn.
Additional origins là kiểm tra độ nhạy thời gian; lựa chọn cuối cần được xác nhận bằng test chưa dùng.
''')
    code(sections['summary'])
    notebook = {'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                 'language_info':{'name':'python','version':'3.12'}},'nbformat':4,'nbformat_minor':4}
    path = root/'m5_tft_window_length_experiment.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print('Built:',path.name,'| cells:',len(cells))


if __name__ == '__main__':
    build()
