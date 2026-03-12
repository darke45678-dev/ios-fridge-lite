/// <reference types="vite/client" />
import { RefObject, useState, useEffect, useRef } from "react";
import { Camera, Loader2, RefreshCw } from "lucide-react";
import { useIngredients } from "../../services/IngredientContext";
import { DetectionSummary } from "../inventory_management/DetectionSummary";
import { notificationService } from "../../services/notificationService";

// 使用 global 宣告來告訴 TypeScript 我們的 ort 在 window 上
declare global {
    interface Window {
        ort: any;
    }
}

interface CameraViewProps {
    videoRef: RefObject<HTMLVideoElement | null>;
}

/**
 * 攝影機掃描視圖 (CameraView - ONNX 離線版)
 */
export function CameraView({ videoRef }: CameraViewProps) {
    const { addItem, tempDetections, clearTempDetections } = useIngredients();
    const [isScanning, setIsScanning] = useState(false);
    const [currentBoxes, setCurrentBoxes] = useState<any[]>([]);
    const [modelLoaded, setModelLoaded] = useState(false);
    const sessionRef = useRef<any>(null);

    // 類別名稱對照表 (由 YOLO 模型訓練時決定)
    const CLASS_NAMES = ["tomato", "spinach", "egg", "eggplant", "rotten"];

    // 初始化：加載 ONNX 模型
    useEffect(() => {
        async function initModel() {
            try {
                // 從 window 取得全域的 ort 引擎
                const ort = window.ort;
                if (!ort) {
                    console.error("❌ 找不到 AI 引擎元件，請確認 index.html 是否正確引入 ort.min.js");
                    return;
                }

                // 從環境取得基礎路徑
                const baseUrl = import.meta.env.BASE_URL || "/";
                const modelUrl = `${baseUrl}best.onnx`;
                
                // 設定 ONNX Runtime WASM 零件經由 CDN 下載以確保版本一致與路徑正確
                const cdnUrl = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/";
                ort.env.wasm.wasmPaths = cdnUrl;
                ort.env.wasm.numThreads = 1;
                ort.env.wasm.proxy = false;

                // 載入模型 (優先嘗試 WebGL GPU 加速)
                const session = await ort.InferenceSession.create(modelUrl, {
                    executionProviders: ["webgl", "wasm"], 
                    graphOptimizationLevel: "all" 
                });
                
                sessionRef.current = session;
                setModelLoaded(true);
                console.log("✅ AI 大腦載入成功！");
            } catch (e) {
                console.error("❌ 模型載入失敗，具體錯誤內容:", e);
                if (e instanceof Error) {
                    console.error("錯誤訊息:", e.message);
                }
            }
        }
        initModel();
    }, []);

    const handleScan = async () => {
        if (!videoRef.current || !sessionRef.current) return;
        setIsScanning(true);
        setCurrentBoxes([]);

        try {
            // 1. 擷取畫面並縮放至 640x640 (YOLO 標準尺寸)
            const canvas = document.createElement("canvas");
            canvas.width = 640;
            canvas.height = 640;
            const ctx = canvas.getContext("2d");
            if (!ctx) return;
            ctx.drawImage(videoRef.current, 0, 0, 640, 640);

            // 2. 影像前處理 (Image to Tensor)
            const imgData = ctx.getImageData(0, 0, 640, 640);
            const input = new Float32Array(3 * 640 * 640);
            for (let i = 0; i < 640 * 640; i++) {
                input[i] = imgData.data[i * 4] / 255.0; // R
                input[i + 640 * 640] = imgData.data[i * 4 + 1] / 255.0; // G
                input[i + 2 * 640 * 640] = imgData.data[i * 4 + 2] / 255.0; // B
            }
            const tensor = new window.ort.Tensor("float32", input, [1, 3, 640, 640]);

            // 3. 執行推理 (Run Inference)
            const feeds = { [sessionRef.current.inputNames[0]]: tensor };
            const results = await sessionRef.current.run(feeds);
            const output = results[sessionRef.current.outputNames[0]].data as Float32Array;

            // 4. 解析輸出 (Post-processing - Simplified YOLOv8 parser)
            // YOLOv8 輸出格式通常為 [1, 84, 8400] (box=4 + classes=80)
            const detections: any[] = [];
            const CONF_THRESHOLD = 0.25;

            // 這裡進行簡易的解碼演算法
            for (let i = 0; i < 8400; i++) {
                let maxConf = 0;
                let classId = -1;

                // 找出該點最大信心的類別 (跳過前 4 個座標值)
                for (let c = 0; c < CLASS_NAMES.length; c++) {
                    const conf = output[8400 * (4 + c) + i];
                    if (conf > maxConf) {
                        maxConf = conf;
                        classId = c;
                    }
                }

                if (maxConf > CONF_THRESHOLD) {
                    const cx = output[i];
                    const cy = output[8400 + i];
                    const w = output[8400 * 2 + i];
                    const h = output[8400 * 3 + i];

                    const x1 = (cx - w / 2) / 640;
                    const y1 = (cy - h / 2) / 640;
                    const x2 = (cx + w / 2) / 640;
                    const y2 = (cy + h / 2) / 640;

                    detections.push({
                        name: CLASS_NAMES[classId],
                        confidence: maxConf,
                        box: [x1, y1, x2, y2],
                        isSpoiled: CLASS_NAMES[classId] === "rotten",
                        category: classId === 1 ? "蔬菜" : "其他"
                    });

                    // 為了範例流暢度，我們先只抓最高信心的一個
                    if (detections.length > 5) break;
                }
            }

            // NMS 模擬：只取最高信心度的結果
            const finalDetections = detections.sort((a, b) => b.confidence - a.confidence).slice(0, 3);

            if (finalDetections.length === 0) {
                notificationService.send("掃描完成", "未偵測到任何食材，請靠近一點或調整角度重試。");
            } else {
                setCurrentBoxes(finalDetections);
                finalDetections.forEach(det => addItem({
                    name: det.name,
                    quantity: 1,
                    category: det.category,
                    confidence: det.confidence,
                    isSpoiled: det.isSpoiled,
                    box: det.box
                }));
            }

        } catch (error) {
            console.warn("AI 核心運作異常:", error);
        } finally {
            setIsScanning(false);
        }
    };

    const handleClear = () => {
        clearTempDetections();
        setCurrentBoxes([]);
    };

    return (
        <div className="flex flex-col items-center w-full max-w-sm">
            <div className="relative w-full">
                {/* AI Status Badge */}
                <div className={`absolute top-4 left-1/2 transform -translate-x-1/2 z-20 bg-[#0f2e24]/80 backdrop-blur-md border ${!modelLoaded ? 'border-red-400' : isScanning ? 'border-amber-400' : 'border-[#00ff88]'} rounded-full px-4 py-1.5 flex items-center gap-2 shadow-[0_0_15px_rgba(0,255,136,0.3)] transition-colors duration-500`}>
                    <div className={`w-2 h-2 rounded-full ${!modelLoaded ? 'bg-red-400' : isScanning ? 'bg-amber-400 animate-pulse' : 'bg-[#00ff88]'} shadow-[0_0_8px_currentColor]`} />
                    <span className={`text-[10px] font-black tracking-widest ${!modelLoaded ? 'text-red-400' : isScanning ? 'text-amber-400' : 'text-[#00ff88]'} uppercase`}>
                        {!modelLoaded ? "Loading Engine..." : isScanning ? "Analyzing..." : "離線 AI 核心已就緒"}
                    </span>
                </div>

                {/* Camera View */}
                <div className="relative aspect-[3/4] bg-[#1a4d3d] rounded-[2.5rem] overflow-hidden border-4 border-[#1a4d3d] shadow-2xl">
                    <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        className="absolute inset-0 w-full h-full object-cover"
                    />

                    {/* Bounding Box Overlay */}
                    <div className="absolute inset-0 z-10 pointer-events-none">
                        {currentBoxes.map((boxData, idx) => boxData.box && (
                            <div
                                key={`box-${idx}`}
                                className="absolute"
                                style={{
                                    left: `${boxData.box[0] * 100}%`,
                                    top: `${boxData.box[1] * 100}%`,
                                    width: `${(boxData.box[2] - boxData.box[0]) * 100}%`,
                                    height: `${(boxData.box[3] - boxData.box[1]) * 100}%`,
                                    borderColor: boxData.isSpoiled ? '#ff4d4d' : '#00ff88',
                                    borderWidth: '2px',
                                    borderStyle: 'solid',
                                    borderRadius: '8px'
                                }}
                            >
                                <div className={`absolute -top-6 left-0 px-2 py-0.5 rounded-t-md text-[8px] font-black uppercase whitespace-nowrap ${boxData.isSpoiled ? 'bg-red-500 text-white' : 'bg-[#00ff88] text-[#0f2e24]'}`}>
                                    {boxData.isSpoiled ? 'BAD' : 'GOOD'} | {boxData.name} | {Math.round((boxData.confidence || 0) * 100)}%
                                </div>
                            </div>
                        ))}
                    </div>

                    <div className="absolute inset-0 bg-gradient-to-b from-[#0f2e24]/40 to-transparent pointer-events-none" />

                    {isScanning && (
                        <div className="absolute inset-0 bg-[#00ff88]/5 flex flex-col items-center justify-center">
                            <div className="w-full h-[2px] bg-amber-400 shadow-[0_0_15px_#fbbf24] absolute top-0 animate-[scan_2s_ease-in-out_infinite]" />
                        </div>
                    )}
                </div>
            </div>

            {tempDetections.length > 0 && (
                <div className="w-full flex justify-end px-2 mb-2 mt-4">
                    <button
                        onClick={handleClear}
                        className="flex items-center gap-2 px-3 py-1.5 bg-red-500/10 text-red-500 rounded-lg border border-red-500/20 hover:bg-red-500 hover:text-white transition-all text-[10px] font-black tracking-widest uppercase"
                    >
                        <RefreshCw size={12} />
                        重新整理畫面
                    </button>
                </div>
            )}

            <DetectionSummary readOnly={true} />

            <div className="w-full mt-8 space-y-3 px-2">
                <button
                    onClick={handleScan}
                    disabled={isScanning || !modelLoaded}
                    className="w-full bg-[#00ff88] text-[#0f2e24] py-4 rounded-2xl font-black text-lg flex items-center justify-center gap-3 hover:bg-[#00dd77] transition-all active:scale-[0.98] shadow-[0_8px_20px_rgba(0,255,136,0.3)] disabled:opacity-50"
                >
                    {isScanning ? <Loader2 size={24} className="animate-spin" /> : <Camera size={24} strokeWidth={3} />}
                    {isScanning ? "LOCAL INFERENCING..." : "離線 AI 辨識"}
                </button>
            </div>
        </div>
    );
}
