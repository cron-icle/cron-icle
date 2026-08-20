//! D3D11 capture session: create the session, pull one frame, read it back
//! to a CPU-side PNG.

pub struct D3d11CaptureSession {
    pub frame_pool: windows::Graphics::Capture::Direct3D11CaptureFramePool,
    pub session: windows::Graphics::Capture::GraphicsCaptureSession,
    pub device: windows::Win32::Graphics::Direct3D11::ID3D11Device,
    pub context: windows::Win32::Graphics::Direct3D11::ID3D11DeviceContext,
}

pub fn initialize(window_handle: isize) -> Result<D3d11CaptureSession, String> {
    use windows::core::Interface;
    use windows::Graphics::Capture::{Direct3D11CaptureFramePool, GraphicsCaptureItem};
    use windows::Graphics::DirectX::{Direct3D11::IDirect3DDevice, DirectXPixelFormat};
    use windows::Win32::Foundation::HMODULE;
    use windows::Win32::Graphics::Direct3D::{D3D_DRIVER_TYPE_HARDWARE, D3D_FEATURE_LEVEL_11_0};
    use windows::Win32::Graphics::Direct3D11::{
        D3D11CreateDevice, ID3D11Device, D3D11_CREATE_DEVICE_FLAG,
    };
    use windows::Win32::Graphics::Dxgi::IDXGIDevice;
    use windows::Win32::System::WinRT::Direct3D11::CreateDirect3D11DeviceFromDXGIDevice;
    use windows::UI::WindowId;
    let item = GraphicsCaptureItem::TryCreateFromWindowId(WindowId {
        Value: window_handle as u64,
    })
    .map_err(|error| format!("unable to create capture item: {error}"))?;
    let mut device: Option<ID3D11Device> = None;
    let mut context: Option<windows::Win32::Graphics::Direct3D11::ID3D11DeviceContext> = None;
    unsafe {
        D3D11CreateDevice(
            None,
            D3D_DRIVER_TYPE_HARDWARE,
            HMODULE::default(),
            D3D11_CREATE_DEVICE_FLAG(0),
            Some(&[D3D_FEATURE_LEVEL_11_0]),
            7,
            Some(&mut device),
            None,
            Some(&mut context),
        )
        .map_err(|error| format!("unable to create D3D11 device: {error}"))?;
    }
    let device = device.ok_or("D3D11 returned no device")?;
    let context = context.ok_or("D3D11 returned no immediate context")?;
    let dxgi_device: IDXGIDevice = device
        .cast()
        .map_err(|error| format!("unable to cast DXGI device: {error}"))?;
    let inspectable = unsafe {
        CreateDirect3D11DeviceFromDXGIDevice(&dxgi_device)
            .map_err(|error| format!("unable to create WinRT Direct3D device: {error}"))?
    };
    let direct3d_device: IDirect3DDevice = inspectable
        .cast()
        .map_err(|error| format!("unable to cast WinRT Direct3D device: {error}"))?;
    let frame_pool = Direct3D11CaptureFramePool::CreateFreeThreaded(
        &direct3d_device,
        DirectXPixelFormat::B8G8R8A8UIntNormalized,
        2,
        item.Size()
            .map_err(|error| format!("unable to read capture size: {error}"))?,
    )
    .map_err(|error| format!("unable to create capture frame pool: {error}"))?;
    let session = frame_pool
        .CreateCaptureSession(&item)
        .map_err(|error| format!("unable to create capture session: {error}"))?;
    session
        .StartCapture()
        .map_err(|error| format!("unable to start capture session: {error}"))?;
    Ok(D3d11CaptureSession {
        frame_pool,
        session,
        device,
        context,
    })
}

pub fn capture_one_frame_png(window_handle: isize) -> Result<Vec<u8>, String> {
    use std::{thread, time::Duration};

    let session = initialize(window_handle)?;
    let mut last_error = String::from("no frame was available");
    for _ in 0..8 {
        match session.capture_next_frame_png() {
            Ok(image) => return Ok(image),
            Err(error) => last_error = error,
        }
        thread::sleep(Duration::from_millis(25));
    }
    Err(last_error)
}

impl D3d11CaptureSession {
    pub fn capture_next_frame_png(&self) -> Result<Vec<u8>, String> {
        use windows::core::Interface;
        use windows::Graphics::DirectX::Direct3D11::IDirect3DSurface;
        use windows::Win32::Graphics::Direct3D11::{
            ID3D11Texture2D, D3D11_CPU_ACCESS_READ, D3D11_MAP_READ, D3D11_TEXTURE2D_DESC,
            D3D11_USAGE_STAGING,
        };
        use windows::Win32::System::WinRT::Direct3D11::IDirect3DDxgiInterfaceAccess;
        let frame = self
            .frame_pool
            .TryGetNextFrame()
            .map_err(|error| format!("unable to acquire capture frame: {error}"))?;
        let surface: IDirect3DSurface = frame
            .Surface()
            .map_err(|error| format!("unable to access capture surface: {error}"))?;
        let access: IDirect3DDxgiInterfaceAccess = surface
            .cast()
            .map_err(|error| format!("unable to access DXGI surface: {error}"))?;
        let texture: ID3D11Texture2D = unsafe {
            access
                .GetInterface()
                .map_err(|error| format!("unable to access D3D11 texture: {error}"))?
        };
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        unsafe {
            texture.GetDesc(&mut desc);
        }
        desc.Usage = D3D11_USAGE_STAGING;
        desc.BindFlags = 0;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ.0 as u32;
        desc.MiscFlags = 0;
        let mut staging = None;
        unsafe {
            self.device
                .CreateTexture2D(&desc, None, Some(&mut staging))
                .map_err(|error| format!("unable to create CPU staging texture: {error}"))?;
        }
        let staging = staging.ok_or("D3D11 returned no staging texture")?;
        unsafe {
            self.context.CopyResource(&staging, &texture);
        }
        let mut mapped = windows::Win32::Graphics::Direct3D11::D3D11_MAPPED_SUBRESOURCE::default();
        unsafe {
            self.context
                .Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
                .map_err(|error| format!("unable to map captured frame: {error}"))?;
        }
        let mut pixels = vec![0u8; (desc.Width as usize) * (desc.Height as usize) * 4];
        unsafe {
            for row in 0..desc.Height as usize {
                let source = core::slice::from_raw_parts(
                    (mapped.pData as *const u8).add(row * mapped.RowPitch as usize),
                    desc.Width as usize * 4,
                );
                pixels[row * desc.Width as usize * 4..(row + 1) * desc.Width as usize * 4]
                    .copy_from_slice(source);
            }
            self.context.Unmap(&staging, 0);
        }
        for pixel in pixels.chunks_exact_mut(4) {
            pixel.swap(0, 2);
            pixel[3] = 255;
        }
        crate::windows_active_window_screenshot::encode_png_rgba(desc.Width, desc.Height, &pixels)
    }
}
