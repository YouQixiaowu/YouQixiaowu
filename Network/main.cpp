#include "asio.hpp"
#include <iostream>
#include <string>
#include <memory>
/*
asio 异步定时器
*/
void Print(asio::error_code ec) {
	std::cout << "Hello, world!" << std::endl;
}
int MainAsynchronousTimer(int argc, char* argv[]) {
	// io上下文：代表操作系统的io操作
	asio::io_context ioc;
	// io对象：3s的定时器
	asio::steady_timer timer(ioc, std::chrono::seconds(3));
	// 定时器的任务
	timer.async_wait(&Print);
	// 执行任务
	ioc.run();
	return 0;
}
/*
asio 同步Echo程序服务端代码
*/
void SynchronousService(asio::ip::tcp::socket socket)
{
	try {
		while (true) {
			std::error_code ec;
			std::string data;
			data.resize(1024);
			socket.read_some(asio::buffer(data), ec);
			if (ec == asio::error::eof) {
				break;
			}
			else if (ec) {
				throw asio::system_error(ec);
			}
			std::cout << data << std::endl;
			asio::write(socket, asio::buffer(data, data.size()));
		}
	}
	catch (const std::exception& e) {
		std::cerr << "Exception: " << e.what() << std::endl;
	}
}
int MainSynchronousSession(int argc, char* argv[]) {
	asio::io_context ioc;
	asio::ip::tcp::acceptor acceptor(ioc, asio::ip::tcp::endpoint(asio::ip::tcp::v4(), 53945));
	while (true)
		SynchronousService(acceptor.accept());
	return 0;
}
/*
asio 异步Echo程序服务端代码
*/
class AsynchronousSession :
	public std::enable_shared_from_this<AsynchronousSession> {
	struct Private { explicit Private() = default; };
public:
	AsynchronousSession(Private, asio::ip::tcp::socket socket) :
		socket_(std::move(socket)) {
		buffer_.resize(1024);
	}
	static std::shared_ptr<AsynchronousSession> create(asio::ip::tcp::socket socket)
	{
		return std::make_shared<AsynchronousSession>(Private(), std::move(socket));
	}
	std::shared_ptr<AsynchronousSession> getptr()
	{
		return shared_from_this();
	}
	void Read() {
		auto self(shared_from_this());
		socket_.async_read_some(
			asio::buffer(buffer_),
			[this, self](std::error_code ec, std::size_t length) {
				if (!ec) {
					std::cout << buffer_.data() << std::endl;
					Write(length);
				}
			});
	}
	void Write(std::size_t length) {
		auto self(shared_from_this());
		asio::async_write(
			socket_,
			asio::buffer(buffer_, length),
			[this, self](std::error_code ec, std::size_t length) {
				if (!ec) {
					Read();
				}
			});
	}
private:
	asio::ip::tcp::socket socket_;
	std::vector<char> buffer_;
};
class Server {
public:
	Server(asio::io_context& ioc, std::uint16_t port)
		: acceptor_(ioc, asio::ip::tcp::endpoint(asio::ip::tcp::v4(), port)) {
		Accept();
	}

private:
	void Accept() {
		acceptor_.async_accept(
			[this](std::error_code ec, asio::ip::tcp::socket socket) {
				if (!ec) {
					AsynchronousSession::create(std::move(socket))->Read();
				}
				Accept();
			});
	}
private:
	asio::ip::tcp::acceptor acceptor_;
};
int MainAsynchronousSession(int argc, char* argv[]) {

	asio::io_context ioc;
	Server server(ioc, 53945);
	ioc.run();
	return 0;
}

int main(int argc, char* argv[])
{
	// return MainAsynchronousTimer(argc, argv);
	// return MainSynchronousSession(argc, argv);
	return MainAsynchronousSession(argc, argv);
}