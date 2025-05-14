import axios, { AxiosResponse, AxiosError } from "axios";

// 定义请求参数类型
interface RequestOptions {
    url: string; // 请求地址
    method?: "GET" | "POST" | "PUT" | "DELETE"; // 请求方法，默认为 GET
    data?: any; // 请求体数据（适用于 POST、PUT）
    params?: any; // URL 参数（适用于 GET）
    headers?: Record<string, string>; // 自定义请求头
    timeout?: number; // 超时时间（毫秒），默认 10 秒
}

// 定义响应数据类型
interface ApiResponse<T = any> {
    success: boolean; // 是否成功
    data?: T; // 响应数据
    error?: string; // 错误信息
}

/**
 * 封装的请求工具方法
 * @param options 请求参数
 * @returns Promise<ApiResponse>
 */
export default async function request<T = any>(options: RequestOptions): Promise<ApiResponse<T>> {
    const {
        url,
        method = "GET",
        data = null,
        params = null,
        headers = {},
        timeout = 60000 * 3, // 默认超时时间为 60 秒
    } = options;

    try {
        // 创建 AbortController 用于超时控制
        const controller = new AbortController();
        const signal = controller.signal;

        // 设置超时定时器
        const timeoutId = setTimeout(() => controller.abort(), timeout);

        // 发起请求
        const response: AxiosResponse<T> = await axios({
            url,
            method,
            data,
            params,
            headers: {
                "Content-Type": "application/json", // 默认 Content-Type
                ...headers, // 自定义请求头覆盖默认值
            },
            maxBodyLength: Infinity, // 允许发送大文件
            signal, // 绑定信号以支持超时
        });

        // 清除定时器
        clearTimeout(timeoutId);

        // 检查响应状态码
        if (response.status >= 200 && response.status < 300) {
            return response.data as ApiResponse<T>;
        } else {
            return {
                success: false,
                error: `HTTP Error: ${response.status} - ${response.statusText}`,
            };
        }
    } catch (error) {
        // 处理错误
        if (axios.isAxiosError(error)) {
            const axiosError = error as AxiosError;
            if (axiosError.code === "ECONNABORTED") {
                return {
                    success: false,
                    error: "请求超时，请稍后重试。",
                };
            } else if (axiosError.response) {
                // 服务器返回了非 2xx 的状态码
                const { status, statusText } = axiosError.response;
                return {
                    success: false,
                    error: `HTTP Error: ${status} - ${statusText}`,
                };
            } else if (axiosError.request) {
                // 请求已发出，但没有收到响应
                return {
                    success: false,
                    error: "网络错误，未收到服务器响应。",
                };
            } else {
                // 其他错误
                return {
                    success: false,
                    error: axiosError.message || "未知错误",
                };
            }
        } else {
            // 非 Axios 错误
            return {
                success: false,
                error: (error as Error).message || "未知错误",
            };
        }
    }
}